"""End-to-end test client for the whisper_api service. Run from the Mac:

    python test_client.py --host <L40S_IP> [--port 8888] \
        [--audio path/to.wav] [--transcript "aaj hum Newton ke laws padhenge"] [--job]

--job exercises the async queue (POST /jobs/* -> poll GET /jobs/{id}) instead of
the sync endpoints, plus a cancel check. Without --audio it looks for a wav in
../orpheus_test/outputs/, else synthesizes a 2s tone (structural checks only).
"""

import argparse
import glob
import math
import os
import struct
import sys
import tempfile
import time
import wave

import requests

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def synth_tone_wav(seconds=2.0, freq=440.0, sr=16000):
    path = os.path.join(tempfile.gettempdir(), "whisper_api_test_tone.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(int(seconds * sr))
        )
        w.writeframes(frames)
    return path


def find_audio():
    outputs = os.path.join(os.path.dirname(__file__), "..", "orpheus_test", "outputs")
    wavs = sorted(glob.glob(os.path.join(outputs, "*.wav")))
    if wavs:
        return wavs[0], True
    print("no wav found in orpheus_test/outputs/ - synthesizing a test tone")
    return synth_tone_wav(), False


def check_words(words, duration, expect_monotonic=True):
    check("words non-empty", len(words) > 0, "no words returned")
    if not words:
        return
    check("every word has start/end", all(w["start"] is not None and w["end"] is not None for w in words))
    check("start <= end", all(w["start"] <= w["end"] for w in words))
    if expect_monotonic:
        starts = [w["start"] for w in words]
        check("starts monotonic non-decreasing", all(a <= b for a, b in zip(starts, starts[1:])))
    check(
        "times within [0, duration+0.5]",
        all(0 <= w["start"] and w["end"] <= duration + 0.5 for w in words),
    )


def print_words(words, limit=15):
    for w in words[:limit]:
        mark = "" if w.get("aligned", True) else "  (interpolated)"
        print(f"    {w['start']:7.2f} - {w['end']:7.2f}  {w['score']:.2f}  {w['word']}{mark}")
    if len(words) > limit:
        print(f"    ... {len(words) - limit} more")


def submit_and_poll(base, task, audio_path, data, poll_interval=3, timeout=1800):
    """POST /jobs/<task>, poll GET /jobs/{id} until terminal. Returns result dict or None."""
    with open(audio_path, "rb") as f:
        r = requests.post(f"{base}/jobs/{task}", files={"file": f}, data=data, timeout=600)
    check(f"jobs/{task} submit 200", r.status_code == 200, r.text[:300])
    if r.status_code != 200:
        return None
    sub = r.json()
    job_id = sub["job_id"]
    print(f"  job_id={job_id} queue_position={sub.get('queue_position')}")

    last_stage, deadline = None, time.time() + timeout
    s = {"status": "queued", "stage": None}
    while time.time() < deadline:
        r = requests.get(f"{base}/jobs/{job_id}", params={"include_result": "false"}, timeout=30)
        if r.status_code != 200:
            check(f"jobs/{task} poll 200", False, f"{r.status_code} {r.text[:200]}")
            return None
        s = r.json()
        if s["stage"] != last_stage:
            print(f"  [{s['status']}] stage={s['stage']} progress={s.get('progress')}")
            last_stage = s["stage"]
        if s["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(poll_interval)

    check(f"jobs/{task} finished ok", s["status"] == "done", f"status={s['status']} error={s.get('error')}")
    if s["status"] != "done":
        return None
    r = requests.get(f"{base}/jobs/{job_id}", timeout=60)
    result = r.json().get("result")
    check(f"jobs/{task} result present", result is not None)
    return result


def check_transcribe_result(t, is_speech):
    print(f"  language={t['language']} duration={t['duration']}s")
    print(f"  text: {t['text'][:200]}")
    if t.get("warnings"):
        print(f"  warnings: {t['warnings']}")
    if is_speech:
        check_words(t["words"], t["duration"])
        print_words(t["words"])
    else:
        print("  (tone input - skipping word assertions on transcribe)")


def check_align_result(a, transcript):
    extra = f" anchor_coverage={a['anchor_coverage']}" if "anchor_coverage" in a else ""
    print(f"  language={a['language']} ({a['language_source']}) mode={a.get('mode')}"
          f" aligned {a['n_aligned']}/{a['n_words']} duration={a['duration']}s{extra}")
    if a.get("warnings"):
        print(f"  warnings: {a['warnings']}")
    n_tokens = len(transcript.split())
    check("one word per transcript token", a["n_words"] == n_tokens,
          f"{a['n_words']} words vs {n_tokens} tokens")
    check_words(a["words"], a["duration"])
    print_words(a["words"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--audio")
    ap.add_argument("--transcript", default="hello world this is a test of alignment")
    ap.add_argument("--job", action="store_true", help="use the async jobs API instead of sync")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    if args.audio:
        audio_path, is_speech = args.audio, True
    else:
        audio_path, is_speech = find_audio()
    print(f"audio: {audio_path}\n")

    print(f"GET {base}/health")
    r = requests.get(f"{base}/health", timeout=10)
    check("health 200", r.status_code == 200, r.text[:200])
    h = r.json()
    print(f"  {h}")
    check("cuda true", h.get("cuda") is True)

    if args.job:
        print(f"\nPOST {base}/jobs/transcribe (async)")
        t = submit_and_poll(base, "transcribe", audio_path, {})
        if t:
            check_transcribe_result(t, is_speech)

        print(f"\nPOST {base}/jobs/align (async)  transcript={args.transcript!r}")
        a = submit_and_poll(base, "align", audio_path, {"transcript": args.transcript})
        if a:
            check_align_result(a, args.transcript)

        print(f"\ncancel check: submit + immediate DELETE")
        with open(audio_path, "rb") as f:
            r = requests.post(f"{base}/jobs/align", files={"file": f},
                              data={"transcript": args.transcript}, timeout=600)
        if r.status_code == 200:
            jid = r.json()["job_id"]
            r = requests.delete(f"{base}/jobs/{jid}", timeout=30)
            # either it was still queued (cancelled) or the worker already grabbed it (409)
            check("cancel is coherent", r.status_code in (200, 409), f"got {r.status_code}")
            print(f"  DELETE -> {r.status_code} {r.json()}")

        print(f"\nPOST {base}/jobs/align (empty transcript -> expect 422)")
        with open(audio_path, "rb") as f:
            r = requests.post(f"{base}/jobs/align", files={"file": f},
                              data={"transcript": "  "}, timeout=60)
        check("empty transcript rejected", r.status_code == 422, f"got {r.status_code}")
    else:
        print(f"\nPOST {base}/transcribe")
        with open(audio_path, "rb") as f:
            r = requests.post(f"{base}/transcribe", files={"file": f}, timeout=600)
        check("transcribe 200", r.status_code == 200, r.text[:300])
        if r.status_code == 200:
            check_transcribe_result(r.json(), is_speech)

        print(f"\nPOST {base}/align  transcript={args.transcript!r}")
        with open(audio_path, "rb") as f:
            r = requests.post(f"{base}/align", files={"file": f},
                              data={"transcript": args.transcript}, timeout=600)
        check("align 200", r.status_code == 200, r.text[:300])
        if r.status_code == 200:
            check_align_result(r.json(), args.transcript)

        print(f"\nPOST {base}/align  (empty transcript -> expect 422)")
        with open(audio_path, "rb") as f:
            r = requests.post(f"{base}/align", files={"file": f},
                              data={"transcript": "  "}, timeout=60)
        check("empty transcript rejected", r.status_code == 422, f"got {r.status_code}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
