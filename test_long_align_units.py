"""Unit tests for the long-form alignment logic (no GPU needed - runs on the Mac).

    pip install rapidfuzz indic-transliteration
    python test_long_align_units.py
"""

import random
import sys

from long_align import (
    MAX_WINDOW_SEC,
    asr_token_times,
    build_windows,
    find_anchors,
    normalize_token,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


print("normalize_token: Hinglish spelling variants + script flips")
PAIRS = [
    ("padhenge", "padenge"),      # dropped aspiration
    ("bhi", "bi"),
    ("samajh", "samaj"),
    ("theek", "thik"),            # ee/i
    ("zaroori", "jaruri"),        # z/j, oo/u, doubled letters
    ("hum", "hum"),
    ("पढ़ेंगे", "padhenge"),        # Devanagari vs romanized
    ("आज", "aaj"),
    ("क्लास", "class"),           # loanword: klas vs clas - allowed to differ, see below
]
for a, b in PAIRS[:-1]:
    ka, kb = normalize_token(a), normalize_token(b)
    check(f"{a!r} ~ {b!r}", ka == kb, f"{ka!r} != {kb!r}")
# loanwords may differ (c vs k) - just require both keys non-empty, anchoring
# tolerates unmatched loanwords between anchors
ka, kb = normalize_token("क्लास"), normalize_token("class")
check("loanword keys non-empty", bool(ka) and bool(kb), f"{ka!r} {kb!r}")
check("punctuation-only -> empty key", normalize_token("...") == "")

print("\nasr_token_times: monotonic-ish, in range")
segs = [
    {"text": "aaj hum newton ke laws", "start": 0.0, "end": 5.0},
    {"text": "padhenge aur samjhenge", "start": 6.0, "end": 10.0},
]
toks, times = asr_token_times(segs)
check("token count", len(toks) == 8)
check("times in range", all(0 <= t <= 10 for t in times))
check("times sorted", times == sorted(times))

print("\nasr_token_times: prefers acoustic word starts, interpolates gaps")
segs_w = [{
    "text": "aaj hum newton ke laws",
    "start": 0.0, "end": 10.0,   # deliberately misleading span
    "words": [
        {"word": "aaj", "start": 2.0, "end": 2.3},
        {"word": "hum", "start": 2.4, "end": 2.6},
        {"word": "newton"},                          # unaligned -> interpolated
        {"word": "ke", "start": 3.4, "end": 3.5},
        {"word": "laws", "start": 3.6, "end": 4.0},
    ],
}]
toks, times = asr_token_times(segs_w)
check("word starts used (not char interp)", times[0] == 2.0 and times[-1] == 3.6, times)
check("missing word interpolated between neighbors", 2.4 < times[2] < 3.4, times[2])
check("word-times monotonic", times == sorted(times))

print("\nfind_anchors: perturbed ASR (10% dropped, 20% respelled)")
random.seed(42)
VOCAB = ("aaj hum ek naya chapter shuru karenge jisme newton ke laws of motion "
         "padhenge fir hum examples dekhenge aur numericals solve karenge "
         "iske baad revision hoga aur test lenge sabse pehle first law "
         "velocity acceleration force mass energy momentum friction gravity "
         "displacement distance speed time graph equation formula derivation "
         "concept doubt question answer board exam important topic samajh gaya "
         "beta dhyan se dekho yahan pe value dalenge toh answer aayega").split()
gt = [random.choice(VOCAB) for _ in range(400)]


def perturb(tok):
    r = random.random()
    if r < 0.10:
        return None  # ASR dropped it
    if r < 0.30:  # spelling variance
        return (tok.replace("aa", "a").replace("ee", "i")
                   .replace("dh", "d").replace("bh", "b"))
    return tok


asr = [p for t in gt for p in [perturb(t)] if p is not None]
asr_times_ = [i * 0.45 for i in range(len(asr))]  # ~0.45s per token
anchors, coverage = find_anchors(gt, asr, asr_times_)
check("coverage > 0.5", coverage > 0.5, f"coverage={coverage:.2f}")
check("anchors found", len(anchors) > 20, f"{len(anchors)}")
idxs, ts = [a[0] for a in anchors], [a[1] for a in anchors]
check("anchor gt indices strictly increasing", all(a < b for a, b in zip(idxs, idxs[1:])))
check("anchor times strictly increasing", all(a < b for a, b in zip(ts, ts[1:])))

print("\nfind_anchors: pathological verbatim-cyclic text degrades but stays usable")
gt_cyc = [VOCAB[i % 33] for i in range(400)]
asr_cyc = [p for t in gt_cyc for p in [perturb(t)] if p is not None]
anchors_c, cov_c = find_anchors(gt_cyc, asr_cyc, [i * 0.45 for i in range(len(asr_cyc))])
check("cyclic coverage still above wrong-file cutoff", cov_c >= 0.15, f"{cov_c:.2f}")
ts_c = [a[1] for a in anchors_c]
check("cyclic anchors monotonic", all(a < b for a, b in zip(ts_c, ts_c[1:])))

print("\nfind_anchors: 3x-verbatim-repeated content must not phase-shift (cycle jump)")
random.seed(7)
sentences = [[random.choice(VOCAB) for _ in range(15)] for _ in range(15)]  # 225-token cycle
gt_rep = [t for _ in range(3) for s in sentences for t in s] + [t for s in sentences[:5] for t in s]
asr_rep, rep_times, t_acc = [], [], 0.0
for tok in gt_rep:
    p = perturb(tok)
    if p is not None:
        asr_rep.append(p)
        rep_times.append(t_acc)
    t_acc += 0.45
gt_true_times = [i * 0.45 for i in range(len(gt_rep))]
anchors_r, cov_r = find_anchors(gt_rep, asr_rep, rep_times)
errs = [abs(at - gt_true_times[gi]) for gi, at in anchors_r]
check("repeated-content coverage > 0.5", cov_r > 0.5, f"{cov_r:.2f}")
check("no anchor off by a cycle (all errors < 20s)",
      max(errs) < 20, f"max_err={max(errs):.1f}s")

print("\nfind_anchors: unrelated transcript -> near-zero coverage")
_, cov_bad = find_anchors(
    "completely different english words about cooking pasta recipes tonight".split(),
    asr, asr_times_,
)
check("unrelated coverage < 0.15", cov_bad < 0.15, f"{cov_bad:.2f}")

print("\nbuild_windows: partition, spans bounded")
duration = len(asr) * 0.45
windows = build_windows(gt, anchors, duration)
check("windows non-empty", len(windows) > 0)
covered = [(w["tok_start"], w["tok_end"]) for w in windows]
check("windows partition tokens exactly",
      covered[0][0] == 0 and covered[-1][1] == len(gt)
      and all(a[1] == b[0] for a, b in zip(covered, covered[1:])), str(covered[:5]))
check("window spans <= MAX_WINDOW_SEC",
      all(w["t1"] - w["t0"] <= MAX_WINDOW_SEC + 0.1 for w in windows))
check("window times monotonic",
      all(a["t1"] <= b["t1"] and a["t0"] <= b["t0"] for a, b in zip(windows, windows[1:])))

print("\nbuild_windows: no anchors at all -> proportional low-confidence windows")
windows = build_windows(gt, [], 1800.0)
check("still partitions", windows[0]["tok_start"] == 0 and windows[-1]["tok_end"] == len(gt))
check("all spans <= MAX", all(w["t1"] - w["t0"] <= MAX_WINDOW_SEC + 0.1 for w in windows))
check("flagged low confidence", all(w["low_confidence"] for w in windows))

print("\nbuild_windows: single token, tiny audio")
windows = build_windows(["hello"], [], 1.0)
check("one window for one token", len(windows) == 1 and windows[0]["tok_end"] == 1)

print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
sys.exit(1 if FAILURES else 0)
