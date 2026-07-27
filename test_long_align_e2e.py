"""E2E test for chunked long-form alignment - no real 2h ground truth needed.

Concatenates short clips with KNOWN per-clip text (repeating them to reach the
target length) into one long wav with silence gaps, so the true time window of
every word is known from the concatenation offsets. Submits a /jobs/align job,
polls, then asserts each word's timestamps land inside its source clip's span.

    python test_long_align_e2e.py --host <IP> [--minutes 20] [--tolerance 0.5] \
        [--clips-dir ../orpheus_test/outputs] [--manifest clips.json]

--manifest: JSON list of {"wav": "path.wav", "text": "spoken text"} overriding
the clips-dir + results.json discovery.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import wave

import requests

FILE_KEYS = ("file", "filename", "wav", "path", "output", "audio")
TEXT_KEYS = ("text", "sentence", "transcript", "prompt")


def load_manifest(args) -> list[dict]:
    """Returns [{"wav": abs_path, "text": str}, ...]."""
    if args.manifest:
        entries = json.load(open(args.manifest))
        base = os.path.dirname(os.path.abspath(args.manifest))
    else:
        rj = os.path.join(args.clips_dir, "results.json")
        if not os.path.exists(rj):
            sys.exit(f"no {rj}; pass --manifest with [{{'wav':..., 'text':...}}] entries")
        data = json.load(open(rj))
        if isinstance(data, dict):  # maybe {"results": [...]} or {filename: text}
            for k in ("results", "items", "outputs"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
            else:
                data = [{"file": k, "text": v} for k, v in data.items() if isinstance(v, str)]
        entries, base = data, args.clips_dir

    clips = []
    for e in entries:
        wav = next((e[k] for k in FILE_KEYS if isinstance(e.get(k), str)), None)
        text = next((e[k] for k in TEXT_KEYS if isinstance(e.get(k), str)), None)
        if not wav or not text:
            continue
        path = wav if os.path.isabs(wav) else os.path.join(base, wav)
        if os.path.exists(path) and path.endswith(".wav"):
            clips.append({"wav": path, "text": " ".join(text.split())})
    if not clips:
        sys.exit("could not resolve any (wav, text) pairs - check results.json keys or use --manifest")
    return clips


def build_concat(clips: list[dict], minutes: float, gap_sec: float):
    """Concatenate clips (cycled) up to `minutes`, return (wav_path, spans, transcript).

    spans[i] = (t_start, t_end, n_tokens) for the i-th concatenated clip.
    """
    with wave.open(clips[0]["wav"], "rb") as w:
        params = w.getparams()
    sr, sw, nch = params.framerate, params.sampwidth, params.nchannels
    gap = b"\x00" * int(gap_sec * sr) * sw * nch

    out_path = os.path.join(tempfile.gettempdir(), "long_align_e2e.wav")
    spans, texts, t = [], [], 0.0
    with wave.open(out_path, "wb") as out:
        out.setnchannels(nch)
        out.setsampwidth(sw)
        out.setframerate(sr)
        i = 0
        while t / 60.0 < minutes:
            clip = clips[i % len(clips)]
            with wave.open(clip["wav"], "rb") as w:
                if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (sr, sw, nch):
                    i += 1
                    continue
                frames = w.readframes(w.getnframes())
                dur = w.getnframes() / sr
            out.writeframes(frames)
            spans.append((t, t + dur, len(clip["text"].split())))
            texts.append(clip["text"])
            t += dur
            out.writeframes(gap)
            t += gap_sec
            i += 1
    return out_path, spans, " ".join(texts), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--clips-dir", default=os.path.join(os.path.dirname(__file__), "..", "orpheus_test", "outputs"))
    ap.add_argument("--manifest")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=0.7)
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--poll", type=float, default=10.0)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    clips = load_manifest(args)
    print(f"{len(clips)} source clips")
    wav_path, spans, transcript, total = build_concat(clips, args.minutes, args.gap)
    n_tokens = len(transcript.split())
    print(f"built {total/60:.1f} min wav, {len(spans)} clip instances, {n_tokens} tokens: {wav_path}")

    print(f"\nsubmitting to {base}/jobs/align ...")
    with open(wav_path, "rb") as f:
        r = requests.post(f"{base}/jobs/align", files={"file": f},
                          data={"transcript": transcript}, timeout=1200)
    if r.status_code != 200:
        sys.exit(f"submit failed: {r.status_code} {r.text[:300]}")
    job_id = r.json()["job_id"]
    print(f"job_id={job_id}")

    last_stage = None
    while True:
        s = requests.get(f"{base}/jobs/{job_id}", params={"include_result": "false"}, timeout=30).json()
        if s["stage"] != last_stage:
            print(f"  [{s['status']}] {s['stage']} ({s.get('progress')})")
            last_stage = s["stage"]
        if s["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(args.poll)
    if s["status"] != "done":
        sys.exit(f"job {s['status']}: {s.get('error')}")

    result = requests.get(f"{base}/jobs/{job_id}", timeout=120).json()["result"]
    words = result["words"]
    print(f"\nmode={result.get('mode')} anchor_coverage={result.get('anchor_coverage')}"
          f" aligned {result['n_aligned']}/{result['n_words']}")
    if result.get("chunk_stats"):
        print(f"chunk_stats: {result['chunk_stats']}")
    for wmsg in result.get("warnings", []):
        print(f"  warning: {wmsg}")

    failures = []
    if len(words) != n_tokens:
        failures.append(f"word count {len(words)} != token count {n_tokens}")

    starts = [w["start"] for w in words]
    if starts != sorted(starts):
        failures.append("starts not monotonic")

    aligned_frac = result["n_aligned"] / max(result["n_words"], 1)
    if aligned_frac < 0.9:
        failures.append(f"aligned fraction {aligned_frac:.2f} < 0.9")

    # each word must land inside its source clip's span (+- tolerance)
    in_window = out_window = 0
    wi = 0
    worst = []
    for (t0, t1, count) in spans:
        for _ in range(count):
            if wi >= len(words):
                break
            w = words[wi]
            mid = (w["start"] + w["end"]) / 2
            if t0 - args.tolerance <= mid <= t1 + args.tolerance:
                in_window += 1
            else:
                out_window += 1
                if len(worst) < 10:
                    worst.append(f"    {w['word']!r} at {w['start']}-{w['end']} expected {t0:.1f}-{t1:.1f}")
            wi += 1
    frac = in_window / max(in_window + out_window, 1)
    print(f"words inside their source-clip window (+-{args.tolerance}s): "
          f"{in_window}/{in_window + out_window} = {frac:.3f}")
    if worst:
        print("  sample misses:")
        print("\n".join(worst))
    if frac < 0.9:
        failures.append(f"in-window fraction {frac:.3f} < 0.9")

    print(f"\n{'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
