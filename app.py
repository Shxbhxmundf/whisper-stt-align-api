"""Whisper STT + forced-alignment API for the L40S.

Sync endpoints (short clips, <= SYNC_MAX_SEC):
  POST /transcribe  - Whisper STT with per-word timestamps (WhisperX align pass)
  POST /align       - forced alignment of a known transcript against audio
Async job queue (long audio, up to ~2h):
  POST /jobs/transcribe, POST /jobs/align  - submit; returns job_id immediately
  GET  /jobs/{id}   - poll status; result included when done
  GET  /jobs        - recent jobs;  DELETE /jobs/{id} - cancel/delete
  GET  /health

Run: ./run.sh (uvicorn, port 8888). See README.md.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager

import torch  # must be imported before whisperx so torch's bundled cuDNN is loaded first
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware

import long_align
from jobs_store import JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper_api")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
PREWARM_ALIGN_LANGS = [
    s.strip() for s in os.environ.get("PREWARM_ALIGN_LANGS", "en,hi").split(",") if s.strip()
]
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
SYNC_MAX_SEC = float(os.environ.get("SYNC_MAX_SEC", "300"))
ALIGN_CHUNKED_ABOVE_SEC = float(os.environ.get("ALIGN_CHUNKED_ABOVE_SEC", "300"))
SINGLE_ALIGN_HARD_MAX_SEC = float(os.environ.get("SINGLE_ALIGN_HARD_MAX_SEC", "420"))
ALIGN_MAX_DURATION_SEC = float(os.environ.get("ALIGN_MAX_DURATION_SEC", "7800"))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "72"))
KEEP_AUDIO = os.environ.get("KEEP_AUDIO", "0") == "1"
JOBS_DIR = os.environ.get(
    "JOBS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
)

DEVICE = "cuda"
SAMPLE_RATE = 16000  # whisperx.load_audio always returns 16 kHz mono


def _git_rev() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


APP_VERSION = _git_rev()

GPU_LOCK = threading.Lock()
WHISPER = None
ALIGN_CACHE: dict = {}  # lang -> (model, metadata)
STORE: JobStore = None
WORK_EVENT = threading.Event()

_noop_cb = lambda stage, progress=0.0: None  # noqa: E731


def detect_script(text: str) -> str:
    """Devanagari transcript -> "hi" align model; Latin (incl. Hinglish) -> "en"."""
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return "hi"
    return "en"


def get_align_model(lang: str):
    """Return (model, metadata) for lang, loading lazily under the GPU lock. None on failure."""
    if lang in ALIGN_CACHE:
        return ALIGN_CACHE[lang]
    try:
        with GPU_LOCK:
            if lang not in ALIGN_CACHE:  # re-check after acquiring
                log.info("loading align model for language=%s", lang)
                ALIGN_CACHE[lang] = whisperx.load_align_model(language_code=lang, device=DEVICE)
        return ALIGN_CACHE[lang]
    except Exception as e:
        log.warning("no align model for language=%s: %s", lang, e)
        return None


def stream_upload(upload: UploadFile, dest_path: str):
    """Stream an upload to disk in chunks - a 2h wav is ~230MB, never hold it in RAM."""
    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with open(dest_path, "wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                os.unlink(dest_path)
                raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_MB}MB limit")
            out.write(chunk)
    if size == 0:
        os.unlink(dest_path)
        raise HTTPException(status_code=400, detail="empty or missing audio file")


def save_upload_tmp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        stream_upload(upload, path)
    except HTTPException:
        raise
    except Exception:
        os.unlink(path)
        raise
    return path


def load_audio_or_400(path: str):
    try:
        audio = whisperx.load_audio(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {e}")
    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="audio decoded to zero samples")
    return audio


def check_duration(duration: float, max_duration: float | None):
    if max_duration is not None and duration > max_duration:
        raise HTTPException(
            status_code=422,
            detail=f"audio is {duration:.0f}s, limit here is {max_duration:.0f}s - "
            "submit long audio via POST /jobs/transcribe or /jobs/align and poll GET /jobs/{id}",
        )


def postprocess_words(segments: list, duration: float):
    """Normalize whisperx word output into a complete, monotonic word list.

    Walks segments[i]["words"] (whisperx's flattened word_segments silently drops
    words that failed to align). Words without timestamps get neighbor-interpolated
    times, score 0.0 and aligned=false. Mutates the word dicts in place so the
    per-segment views stay consistent, and returns (flat_words, warnings).
    """
    words = []
    for seg in segments:
        cleaned = []
        for w in seg.get("words", []):
            cleaned.append(
                {
                    "word": (w.get("word") or "").strip(),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "score": w.get("score"),
                    "aligned": w.get("start") is not None and w.get("end") is not None,
                }
            )
        seg["words"] = cleaned
        words.extend(cleaned)

    warnings = []
    n = len(words)
    n_missing = sum(1 for w in words if not w["aligned"])
    if n_missing:
        i = 0
        while i < n:
            if words[i]["aligned"]:
                i += 1
                continue
            j = i
            while j < n and not words[j]["aligned"]:
                j += 1
            left_t = words[i - 1]["end"] if i > 0 else 0.0
            right_t = words[j]["start"] if j < n else duration
            right_t = max(right_t, left_t)
            span = right_t - left_t
            count = j - i
            for k in range(count):
                words[i + k]["start"] = left_t + span * k / count
                words[i + k]["end"] = left_t + span * (k + 1) / count
            i = j
        warnings.append(
            f"{n_missing}/{n} words had no direct alignment; timestamps interpolated"
        )

    # clamp to [0, duration], enforce end >= start and monotonic starts, round
    prev_start = 0.0
    for w in words:
        w["start"] = min(max(w["start"], 0.0), duration)
        w["end"] = min(max(w["end"], w["start"]), duration)
        if w["start"] < prev_start:
            w["start"] = prev_start
            w["end"] = max(w["end"], w["start"])
        prev_start = w["start"]
        w["start"] = round(w["start"], 3)
        w["end"] = round(w["end"], 3)
        w["score"] = round(w["score"], 3) if w["score"] is not None else 0.0
    return words, warnings


# ---------------------------------------------------------------------------
# Shared inference (called by both the sync endpoints and the job worker)
# ---------------------------------------------------------------------------

def run_transcribe(audio_path: str, language: str | None, align: bool,
                   max_duration: float | None = None, progress_cb=_noop_cb) -> dict:
    audio = load_audio_or_400(audio_path)
    duration = len(audio) / SAMPLE_RATE
    check_duration(duration, max_duration)
    warnings = []

    progress_cb("transcribing", 0.1)
    try:
        with GPU_LOCK:
            result = WHISPER.transcribe(audio, batch_size=BATCH_SIZE, language=language)
    except ValueError as e:  # e.g. unsupported language code
        raise HTTPException(status_code=422, detail=str(e))

    detected = result.get("language") or language or "en"
    segments = result["segments"]

    if align and segments:
        progress_cb("aligning", 0.6)
        am = get_align_model(detected)
        if am is None:
            warnings.append(
                f"no alignment model for language '{detected}'; returning segment-level only"
            )
        else:
            with GPU_LOCK:
                segments = whisperx.align(
                    segments, am[0], am[1], audio, DEVICE, return_char_alignments=False
                )["segments"]

    progress_cb("postprocessing", 0.95)
    words, w_warnings = postprocess_words(segments, duration)
    warnings.extend(w_warnings)

    return {
        "text": " ".join(s["text"].strip() for s in segments).strip(),
        "language": detected,
        "duration": round(duration, 3),
        "model": WHISPER_MODEL,
        "segments": [
            {
                "id": i,
                "start": round(float(s.get("start", 0.0)), 3),
                "end": round(float(s.get("end", duration)), 3),
                "text": s["text"].strip(),
                "words": s.get("words", []),
            }
            for i, s in enumerate(segments)
        ],
        "words": words,
        "warnings": warnings,
    }


def run_align(audio_path: str, transcript: str, language: str | None, mode: str = "auto",
              max_duration: float | None = None, progress_cb=_noop_cb) -> dict:
    transcript_norm = " ".join(transcript.split())
    if not transcript_norm:
        raise HTTPException(status_code=422, detail="transcript is empty")
    if mode not in ("auto", "single", "chunked"):
        raise HTTPException(status_code=422, detail="mode must be auto, single or chunked")

    if language:
        lang, lang_source = language, "explicit"
    else:
        lang, lang_source = detect_script(transcript_norm), "auto_script_detect"

    am = get_align_model(lang)
    if am is None:
        raise HTTPException(
            status_code=422,
            detail=f"no alignment model for language '{lang}' "
            f"(loaded: {sorted(ALIGN_CACHE)}; Latin/Hinglish -> en, Devanagari -> hi)",
        )

    audio = load_audio_or_400(audio_path)
    duration = len(audio) / SAMPLE_RATE
    check_duration(duration, max_duration)

    chunked = mode == "chunked" or (mode == "auto" and duration > ALIGN_CHUNKED_ABOVE_SEC)
    if mode == "single" and duration > SINGLE_ALIGN_HARD_MAX_SEC:
        raise HTTPException(
            status_code=422,
            detail=f"single-segment alignment OOMs beyond ~{SINGLE_ALIGN_HARD_MAX_SEC:.0f}s "
            f"of audio (got {duration:.0f}s) - use mode=chunked or mode=auto",
        )

    anchor_coverage, chunk_stats = None, None
    if chunked:
        segments, warnings, anchor_coverage, chunk_stats = long_align.chunked_align(
            audio, transcript_norm, am[0], am[1], WHISPER, GPU_LOCK,
            BATCH_SIZE, SAMPLE_RATE, progress_cb, get_align_model_fn=get_align_model,
        )
    else:
        progress_cb("aligning", 0.3)
        segments = [{"text": transcript_norm, "start": 0.0, "end": duration}]
        with GPU_LOCK:
            segments = whisperx.align(
                segments, am[0], am[1], audio, DEVICE, return_char_alignments=False
            )["segments"]
        warnings = []

    progress_cb("postprocessing", 0.97)
    words, pp_warnings = postprocess_words(segments, duration)
    warnings.extend(pp_warnings)

    tokens = transcript_norm.split(" ")
    if len(words) != len(tokens):
        warnings.append(
            f"word count mismatch: transcript has {len(tokens)} tokens, "
            f"alignment returned {len(words)} words"
        )

    resp = {
        "language": lang,
        "language_source": lang_source,
        "duration": round(duration, 3),
        "mode": "chunked" if chunked else "single",
        "words": words,
        "n_words": len(words),
        "n_aligned": sum(1 for w in words if w["aligned"]),
        "warnings": warnings,
    }
    if anchor_coverage is not None:
        resp["anchor_coverage"] = round(anchor_coverage, 3)
    if chunk_stats is not None:
        resp["chunk_stats"] = chunk_stats
    return resp


# ---------------------------------------------------------------------------
# Job worker
# ---------------------------------------------------------------------------

def _run_job(job: dict):
    jid = job["id"]
    params = json.loads(job["params"])

    def cb(stage, progress=0.0):
        STORE.update_stage(jid, stage, progress)

    try:
        if job["task"] == "transcribe":
            result = run_transcribe(
                job["audio_path"], params.get("language"), params.get("align", True),
                max_duration=None, progress_cb=cb,
            )
        else:
            result = run_align(
                job["audio_path"], params["transcript"], params.get("language"),
                params.get("mode", "auto"), max_duration=ALIGN_MAX_DURATION_SEC,
                progress_cb=cb,
            )
        STORE.set_duration(jid, result.get("duration"))
        result_path = os.path.join(STORE.job_dir(jid), "result.json")
        with open(result_path, "w") as f:
            json.dump(result, f)
        STORE.finish(jid, result_path=result_path)
        log.info("job %s done (%s, %.0fs audio)", jid, job["task"], result.get("duration", 0))
    except Exception as e:
        detail = getattr(e, "detail", None) or str(e)
        log.error("job %s failed:\n%s", jid, traceback.format_exc())
        STORE.finish(jid, error=str(detail))
    finally:
        if not KEEP_AUDIO and job["audio_path"] and os.path.exists(job["audio_path"]):
            os.unlink(job["audio_path"])


def _worker_loop():
    last_purge = 0.0
    while True:
        try:
            if time.time() - last_purge > 3600:
                STORE.purge_older_than(JOB_TTL_HOURS)
                last_purge = time.time()
            job = STORE.claim_next()
            if job is None:
                WORK_EVENT.wait(timeout=30.0)
                WORK_EVENT.clear()
                continue
            _run_job(job)
        except Exception:
            log.error("worker loop error:\n%s", traceback.format_exc())
            time.sleep(5)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global WHISPER, STORE
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False - CPU torch got installed or the venv "
            "is broken. Re-run setup_l40s.sh (torch must come from the cu12x index)."
        )
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info("loading whisper model %s (%s) ...", WHISPER_MODEL, COMPUTE_TYPE)
    WHISPER = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
    for lang in PREWARM_ALIGN_LANGS:
        get_align_model(lang)
    STORE = JobStore(JOBS_DIR)
    STORE.recover_on_boot()
    threading.Thread(target=_worker_loop, daemon=True, name="job-worker").start()
    log.info("startup complete: whisper=%s align=%s jobs=%s",
             WHISPER_MODEL, list(ALIGN_CACHE), STORE.counts())
    yield


app = FastAPI(title="whisper_api", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.cuda.is_available(),
        "whisper_model": WHISPER_MODEL,
        "align_models_loaded": sorted(ALIGN_CACHE),
        "busy": GPU_LOCK.locked(),
        "jobs": STORE.counts() if STORE else {},
    }


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    align: bool = Form(True),
):
    tmp_path = save_upload_tmp(file)
    try:
        return run_transcribe(tmp_path, language, align, max_duration=SYNC_MAX_SEC)
    except HTTPException:
        raise
    except Exception as e:
        log.error("transcribe failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")
    finally:
        os.unlink(tmp_path)


@app.post("/align")
def align_endpoint(
    file: UploadFile = File(...),
    transcript: str = Form(...),
    language: str | None = Form(None),
    mode: str = Form("auto"),
):
    tmp_path = save_upload_tmp(file)
    try:
        return run_align(tmp_path, transcript, language, mode, max_duration=SYNC_MAX_SEC)
    except HTTPException:
        raise
    except Exception as e:
        log.error("align failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"alignment failed: {e}")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Jobs API
# ---------------------------------------------------------------------------

def _submit_job(task: str, file: UploadFile, params: dict) -> dict:
    job_id = STORE.new_id()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    audio_path = os.path.join(STORE.job_dir(job_id), f"input{suffix}")
    try:
        stream_upload(file, audio_path)
    except Exception:
        shutil.rmtree(STORE.job_dir(job_id), ignore_errors=True)
        raise
    params["filename"] = file.filename
    STORE.enqueue(job_id, task, params, audio_path)
    WORK_EVENT.set()
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": STORE.queue_position(job_id),
        "poll": f"/jobs/{job_id}",
    }


@app.post("/jobs/transcribe")
def submit_transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    align: bool = Form(True),
):
    return _submit_job("transcribe", file, {"language": language, "align": align})


@app.post("/jobs/align")
def submit_align(
    file: UploadFile = File(...),
    transcript: str = Form(...),
    language: str | None = Form(None),
    mode: str = Form("auto"),
):
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="transcript is empty")
    if mode not in ("auto", "single", "chunked"):
        raise HTTPException(status_code=422, detail="mode must be auto, single or chunked")
    return _submit_job(
        "align", file, {"transcript": transcript, "language": language, "mode": mode}
    )


@app.get("/jobs")
def list_jobs(limit: int = 50):
    return {"jobs": STORE.list_jobs(min(limit, 200))}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, include_result: bool = True):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job id")
    resp = {
        "job_id": job["id"],
        "task": job["task"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "audio_duration": job["audio_duration"],
        "error": job["error"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
    }
    if job["status"] == "queued":
        resp["queue_position"] = STORE.queue_position(job_id)
    if job["status"] == "done" and include_result:
        try:
            with open(job["result_path"]) as f:
                resp["result"] = json.load(f)
        except Exception as e:
            resp["error"] = f"result file unavailable: {e}"
    return resp


@app.delete("/jobs/{job_id}")
def cancel_or_delete_job(job_id: str):
    job = STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job["status"] == "queued":
        STORE.cancel(job_id)
        return {"job_id": job_id, "status": "cancelled"}
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="job is running and cannot be cancelled")
    STORE.delete(job_id)
    return {"job_id": job_id, "deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8888")))
