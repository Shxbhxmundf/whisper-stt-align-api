# whisper_api — Whisper STT + forced alignment on the L40S

One FastAPI service for QA-ing Hinglish TTS output and processing long lectures:

- **`POST /transcribe`** / **`POST /align`** — synchronous, for short clips (≤ `SYNC_MAX_SEC`, default 5 min).
- **`POST /jobs/transcribe`** / **`POST /jobs/align`** — async job queue for long audio (up to ~2h): submit, get a `job_id` back immediately, poll `GET /jobs/{id}`.
- **`GET /health`** — liveness, `busy` flag, queue counts.

STT = WhisperX (faster-whisper `large-v3`, float16) + wav2vec2 word-timestamp pass. `/align` does forced alignment of a **known transcript** (your TTS input / lecture script) → per-word start/end/score; long files are handled by an ASR-anchored chunked pipeline (single-segment wav2vec2 alignment OOMs past ~7 min — the service routes automatically).

## Quickstart on the L40S

```bash
# from the Mac
scp -P <PORT> -r whisper_api <USER>@<IP>:~/

# on the box
cd ~/whisper_api
./setup_l40s.sh              # or: ./setup_l40s.sh cu124   (match nvidia-smi CUDA version)
./setup_l40s.sh --prefetch   # optional: download all models (~4GB) now
tmux new -s whisper_api 'while true; do ./run.sh; sleep 2; done'   # auto-restart on crash
```

First startup without `--prefetch` downloads ~4GB from HF (no token needed — all models are ungated). Port **8888** must be reachable, or tunnel from the Mac: `ssh -p <PORT> -L 8888:localhost:8888 <USER>@<IP>` and use `http://localhost:8888`.

## Sync usage (short clips)

```bash
curl -s http://<IP>:8888/health

curl -s -F "file=@sample.wav" http://<IP>:8888/transcribe | python3 -m json.tool
curl -s -F "file=@sample.wav" -F "language=hi" -F "align=false" http://<IP>:8888/transcribe

# forced alignment of known TTS text (Hinglish / Latin script -> en align model)
curl -s -F "file=@tts_out.wav" -F "transcript=aaj hum Newton ke laws padhenge" \
  http://<IP>:8888/align | python3 -m json.tool

# Devanagari auto-routes to the hi align model
curl -s -F "file=@tts_out.wav" -F "transcript=आज हम पढ़ेंगे" http://<IP>:8888/align
```

Audio longer than `SYNC_MAX_SEC` gets a 422 pointing you at the jobs API.

## Jobs API (long audio)

```bash
# submit (returns immediately)
curl -s -F "file=@lecture.mp3" http://<IP>:8888/jobs/transcribe
# -> {"job_id": "a1b2c3d4e5f6", "status": "queued", "queue_position": 1, "poll": "/jobs/a1b2c3d4e5f6"}

curl -s -F "file=@lecture.mp3" -F "transcript=$(cat lecture_script.txt)" http://<IP>:8888/jobs/align

# poll (5-15s interval is plenty; result is included once status=done)
curl -s http://<IP>:8888/jobs/a1b2c3d4e5f6 | python3 -m json.tool
curl -s "http://<IP>:8888/jobs/a1b2c3d4e5f6?include_result=false"   # cheap status-only poll

curl -s http://<IP>:8888/jobs                # recent jobs
curl -s -X DELETE http://<IP>:8888/jobs/<id> # cancel (queued) / delete (finished); 409 if running
```

Job status: `queued → running (stage: transcribing / anchoring / aligning window i/N / postprocessing) → done | failed`. A 2h file takes roughly 6–12 min. Notes:

- **One GPU, FIFO**: jobs run one at a time; `queue_position` tells you how many are ahead. `/health` stays responsive during jobs (`busy: true`).
- **Crash-safe**: jobs live in SQLite (`jobs/jobs.db`). A job interrupted by a restart is requeued once, then failed. Results persist for `JOB_TTL_HOURS` (72h default) — fetch and store them; input audio is deleted right after processing (`KEEP_AUDIO=1` to keep).
- **Cancellation** only works while queued — a running GPU job can't be killed safely.
- Prefer **mp3/m4a uploads** for long audio (~115MB for 2h vs ~230MB wav) — everything is resampled to 16kHz mono anyway. Upload cap: `MAX_UPLOAD_MB` (500).

## Long-form alignment (how /align handles 2h)

Single-segment wav2vec2 alignment OOMs beyond ~7 minutes (attention is O(frames²)). Above `ALIGN_CHUNKED_ABOVE_SEC` (300s) the service switches to a chunked pipeline: Whisper transcribes the audio for rough token times → the ground-truth transcript is anchor-matched to the ASR stream on normalized phonetic keys (handles Hinglish spelling variance like padhenge/padenge and Whisper emitting Devanagari against a Latin script) → the file is cut into ≤4 min windows at anchors → each window force-aligned → stitched. Response adds `mode: "chunked"` and `anchor_coverage` (0–1; below 0.15 the transcript probably doesn't match the audio — all words come back unaligned with a warning instead of garbage). Force a path with `mode=single|chunked` if needed.

Word objects everywhere are `{"word", "start", "end", "score", "aligned"}` (seconds, 3dp). Words that can't be aligned directly (digits, punctuation-only, wrong-script tokens, low-anchor regions) are **not dropped**: they get neighbor-interpolated timestamps, `score: 0.0`, `aligned: false`, and a warning — `/align` always returns exactly one word per whitespace token of the transcript, with monotonically non-decreasing starts.

## Language semantics (important)

Alignment models are per-language **and per-script** (character-vocab based). Wrong pairing = total alignment failure, so:

| Transcript script | Align model | Notes |
|---|---|---|
| Latin (English **and romanized Hinglish**) | `en` (torchaudio wav2vec2) | default when no Devanagari chars found |
| Devanagari | `hi` (`theainerd/Wav2Vec2-large-xlsr-hindi`) | auto-selected if any U+0900–U+097F char present |

`/align` auto-detects script from the transcript (`language_source: auto_script_detect`); pass `language=en|hi` to override. `/transcribe` aligns with the model for Whisper's detected/passed language.

## Config (env vars)

| Var | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `large-v3-turbo` = 2–5x faster, slightly worse on Hindi/Hinglish |
| `PORT` | `8888` | |
| `COMPUTE_TYPE` | `float16` | |
| `BATCH_SIZE` | `8` | transcribe batch size |
| `PREWARM_ALIGN_LANGS` | `en,hi` | align models loaded at startup; others lazy-load |
| `MAX_UPLOAD_MB` | `500` | |
| `SYNC_MAX_SEC` | `300` | sync endpoints reject longer audio (use /jobs) |
| `ALIGN_CHUNKED_ABOVE_SEC` | `300` | /align switches to the chunked pipeline above this |
| `ALIGN_MAX_DURATION_SEC` | `7800` | hard cap for align jobs |
| `JOB_TTL_HOURS` | `72` | finished jobs (rows + files) purged after this |
| `KEEP_AUDIO` | `0` | `1` keeps uploaded audio after the job finishes |
| `JOBS_DIR` | `./jobs` | SQLite + per-job files |

## Testing

```bash
# Mac, no GPU needed - anchoring/windowing logic
pip install rapidfuzz indic-transliteration && python test_long_align_units.py

# against the box: sync endpoints, then the jobs flow
python test_client.py --host <IP> --audio ../orpheus_test/outputs/some.wav --transcript "text spoken"
python test_client.py --host <IP> --audio ... --transcript "..." --job

# chunked long-align correctness: concatenates known clips into a 20min wav,
# checks every word lands in its source clip's time window
python test_long_align_e2e.py --host <IP> --minutes 20
```

## Troubleshooting

- **`torch.cuda.is_available() False` / everything on CPU** — torch got installed from PyPI instead of the CUDA index (installing whisperx before torch does this). Delete `.venv`, re-run `./setup_l40s.sh` with the right `cuXXX` tag. Never use `--system-site-packages`.
- **`libcudnn*.so` load errors** — shouldn't happen with these pins (ctranslate2 ≥4.6.3 doesn't require cuDNN), but `run.sh` also puts the venv's `nvidia/*/lib` dirs on `LD_LIBRARY_PATH`. Don't mix an apt-installed cuDNN with the venv one.
- **mp3 upload fails** — install ffmpeg (`sudo apt-get install -y ffmpeg`).
- **`externally-managed-environment` (PEP 668)** — you're pip-installing outside the venv; `source .venv/bin/activate` first.
- **First request slow** — CUDA context warmup; model downloads are avoided via `--prefetch` but the first inference is still slower.
- **Job stuck in `queued`** — check the server log in tmux; the worker thread logs every claim/finish. `GET /health` shows queue counts.
- **`anchor_coverage` very low on an align job** — the transcript likely doesn't match the audio (wrong file/script); words come back `aligned:false` rather than misleadingly placed.
