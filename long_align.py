"""Chunked long-form forced alignment.

Single-segment wav2vec2 alignment OOMs past ~7 minutes of audio (attention is
O(frames^2), the trellis O(frames x chars)), so long files are handled by:
ASR pass for rough token times -> anchor-match the ground-truth transcript to
the ASR stream on normalized phonetic keys -> cut the file into <=4 min windows
at anchors -> whisperx.align each window -> stitch in transcript order.

Heavy imports (torch/whisperx) are function-local so the anchoring/windowing
logic stays unit-testable on a machine without GPU deps.
"""

import bisect
import difflib
import logging
import math
import re
import unicodedata
from collections import Counter

from indic_transliteration import sanscript
from rapidfuzz import fuzz

log = logging.getLogger("whisper_api.long_align")

MIN_ANCHOR_TOKENS = 3
ANCHOR_FUZZ_MIN = 70
MIN_ANCHOR_COVERAGE = 0.15
ANCHOR_BLOCK_TOKENS = 60    # GT tokens matched per band
ANCHOR_SLACK_TOKENS = 60    # ASR tokens of slack around the mapped position
ANCHOR_NGRAM = 4            # n-gram size for unique (repetition-proof) anchors
TARGET_WINDOW_SEC = 60.0
MAX_WINDOW_SEC = 240.0  # safely under the ~7 min single-segment OOM wall
WINDOW_PAD_SEC = 0.5

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_POST_CONSONANT_H = re.compile(r"([bcdfgjklmnpqrstvxy])h")
_REPEATS = re.compile(r"(.)\1+")


def normalize_token(tok: str) -> str:
    """Collapse a GT or ASR token to a coarse phonetic key for anchor matching.

    Handles Hinglish spelling variance (padhenge/padenge, bhi/bi) and script
    mismatch (Whisper may emit Devanagari against a Latin ground truth) by
    transliterating and then applying a crude Hinglish soundex. Keys only need
    to be stable across both streams, not linguistically pretty.
    """
    if _DEVANAGARI.search(tok):
        tok = sanscript.transliterate(tok, sanscript.DEVANAGARI, sanscript.ITRANS)
    tok = unicodedata.normalize("NFD", tok)
    tok = "".join(c for c in tok if not unicodedata.combining(c))
    tok = _NON_ALNUM.sub("", tok.lower())
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ph", "f"), ("w", "v"), ("z", "j")):
        tok = tok.replace(a, b)
    tok = _POST_CONSONANT_H.sub(r"\1", tok)
    tok = tok.replace("c", "k")  # conflate c/k (class/klas, ch->c->k)
    tok = tok.replace("m", "n")  # conflate nasals (anusvara vs n/m spellings)
    tok = _REPEATS.sub(r"\1", tok)
    if len(tok) > 2 and tok.endswith("a"):
        tok = tok[:-1]  # Hindi schwa deletion: transliterated Aja vs spoken/written aaj
    return tok


def asr_token_times(asr_segments: list[dict]) -> tuple[list[str], list[float]]:
    """Flatten ASR segments to (tokens, estimated time per token).

    Token time = segment start + linear interpolation of the token's character
    midpoint within the segment text. +-1-2s accuracy, plenty for windowing.
    """
    tokens, times = [], []
    for seg in asr_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        span = max(end - start, 0.01)
        pos = 0
        for tok in text.split():
            mid = pos + len(tok) / 2
            tokens.append(tok)
            times.append(start + span * mid / max(len(text), 1))
            pos += len(tok) + 1
    return tokens, times


def _lis(points: list[tuple], value) -> list[tuple]:
    """Longest subsequence of `points` (already sorted by primary coord) with
    strictly increasing value(point). O(n log n). Keeps the maximum number of
    mutually consistent anchors instead of greedily locking onto an early outlier."""
    if not points:
        return []
    tails, tails_at, parent = [], [], [None] * len(points)
    for i, p in enumerate(points):
        v = value(p)
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tails_at.append(i)
        else:
            tails[k] = v
            tails_at[k] = i
        parent[i] = tails_at[k - 1] if k > 0 else None
    out, i = [], tails_at[-1]
    while i is not None:
        out.append(points[i])
        i = parent[i]
    return out[::-1]


def _unique_ngram_pairs(gt_keys: list[str], asr_keys: list[str], n: int) -> list[tuple[int, int]]:
    """(gt_idx, asr_idx) for n-grams occurring exactly once in BOTH streams.
    Such anchors cannot be phase-confused by repeated content, by construction."""
    gt_grams = [tuple(gt_keys[i:i + n]) for i in range(len(gt_keys) - n + 1)]
    asr_grams = [tuple(asr_keys[i:i + n]) for i in range(len(asr_keys) - n + 1)]
    gt_counts, asr_counts = Counter(gt_grams), Counter(asr_grams)
    asr_pos = {g: i for i, g in enumerate(asr_grams) if asr_counts[g] == 1}
    pairs = [
        (i, asr_pos[g])
        for i, g in enumerate(gt_grams)
        if gt_counts[g] == 1 and g in asr_pos
    ]
    return _lis(sorted(pairs), lambda p: p[1])


def _interp_map(pairs: list[tuple[int, int]], n_gt: int, n_asr: int):
    """Piecewise-linear gt_idx -> asr_idx map through sparse anchor pairs,
    extended to the boundaries with the global ratio slope."""
    r = n_asr / max(n_gt, 1)
    if not pairs:
        return lambda i: i * r
    xs = [p[0] for p in pairs]
    ys = [float(p[1]) for p in pairs]
    xs = [0] + xs + [n_gt]
    ys = [max(0.0, ys[0] - pairs[0][0] * r)] + ys + [min(float(n_asr), ys[-1] + (n_gt - pairs[-1][0]) * r)]

    def f(i: float) -> float:
        j = min(max(bisect.bisect_right(xs, i) - 1, 0), len(xs) - 2)
        x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (i - x0) / (x1 - x0)

    return f


def find_anchors(
    gt_tokens: list[str], asr_tokens: list[str], asr_times: list[float]
) -> tuple[list[tuple[int, float]], float]:
    """Match GT tokens to ASR tokens on normalized keys.

    Repetition-safe two-stage matching (lectures repeat phrases verbatim; a
    single global SequenceMatcher phase-shifts onto the wrong occurrence):
      A. unique-n-gram anchors (occur once in both streams -> unambiguous)
         fit a piecewise-linear GT->ASR position map;
      B. SequenceMatcher runs banded around that map with tight slack, so a
         repeat further than ~ANCHOR_SLACK_TOKENS away is unreachable.
    Final points are LIS-filtered to be strictly monotonic in index and time.

    Returns (anchor points [(gt_token_index, audio_time)...], coverage =
    kept fraction of matchable GT tokens).
    """
    gt_keys, gt_map = [], []
    for i, t in enumerate(gt_tokens):
        k = normalize_token(t)
        if k:
            gt_keys.append(k)
            gt_map.append(i)
    asr_keys, asr_map = [], []
    for i, t in enumerate(asr_tokens):
        k = normalize_token(t)
        if k:
            asr_keys.append(k)
            asr_map.append(i)
    if not gt_keys or not asr_keys:
        return [], 0.0

    amap = _interp_map(
        _unique_ngram_pairs(gt_keys, asr_keys, ANCHOR_NGRAM), len(gt_keys), len(asr_keys)
    )

    points = []
    for b0 in range(0, len(gt_keys), ANCHOR_BLOCK_TOKENS):
        b1 = min(b0 + ANCHOR_BLOCK_TOKENS, len(gt_keys))
        lo = max(0, int(amap(b0)) - ANCHOR_SLACK_TOKENS)
        hi = min(len(asr_keys), int(amap(b1)) + ANCHOR_SLACK_TOKENS)
        if hi <= lo:
            continue
        sm = difflib.SequenceMatcher(None, gt_keys[b0:b1], asr_keys[lo:hi], autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            if size < MIN_ANCHOR_TOKENS:
                continue
            a, b = b0 + a, lo + b
            gt_raw = " ".join(gt_tokens[gt_map[a + k]] for k in range(size))
            asr_raw = " ".join(asr_tokens[asr_map[b + k]] for k in range(size))
            if fuzz.ratio(gt_raw.lower(), asr_raw.lower()) < ANCHOR_FUZZ_MIN:
                continue
            for k in range(size):
                points.append((gt_map[a + k], asr_times[asr_map[b + k]]))

    points.sort(key=lambda p: p[0])
    kept = _lis(points, lambda p: p[1])
    return kept, len(kept) / len(gt_keys)


def build_windows(
    gt_tokens: list[str],
    anchors: list[tuple[int, float]],
    duration: float,
    target_sec: float = TARGET_WINDOW_SEC,
    max_sec: float = MAX_WINDOW_SEC,
) -> list[dict]:
    """Partition the token sequence into time windows cut at anchors.

    Every token lands in exactly one window; window spans stay <= max_sec by
    proportionally splitting anchor gaps (flagged low_confidence).
    """
    n = len(gt_tokens)
    if n == 0:
        return []
    pts = [(0, 0.0)]
    for idx, t in anchors:
        if pts[-1][0] < idx < n and pts[-1][1] < t < duration:
            pts.append((idx, t))
    pts.append((n, max(duration, pts[-1][1] + 0.05)))

    cuts = [pts[0]]
    for p in pts[1:-1]:
        if p[1] - cuts[-1][1] >= target_sec:
            cuts.append(p)
    cuts.append(pts[-1])

    windows = []
    for (i0, t0), (i1, t1) in zip(cuts, cuts[1:]):
        if i1 <= i0:
            continue
        t1 = max(t1, t0 + 0.05)
        span = t1 - t0
        if span <= max_sec:
            windows.append({"tok_start": i0, "tok_end": i1, "t0": t0, "t1": t1,
                            "low_confidence": False})
            continue
        # anchor gap too wide: split by char proportion, linear times
        k = math.ceil(span / target_sec)
        toks = gt_tokens[i0:i1]
        char_cum, total = [], sum(len(t) + 1 for t in toks)
        acc = 0
        for t in toks:
            acc += len(t) + 1
            char_cum.append(acc)
        prev_i, prev_t = i0, t0
        for piece in range(1, k + 1):
            if piece == k:
                cut_i, cut_t = i1, t1
            else:
                frac = piece / k
                cut_i = i0 + next(
                    (j + 1 for j, c in enumerate(char_cum) if c >= frac * total),
                    len(toks),
                )
                cut_i = min(max(cut_i, prev_i + 1), i1 - (k - piece))
                cut_t = t0 + span * frac
            if cut_i > prev_i:
                windows.append({"tok_start": prev_i, "tok_end": cut_i,
                                "t0": prev_t, "t1": cut_t, "low_confidence": True})
                prev_i, prev_t = cut_i, cut_t
    return windows


def _bare_words(tokens: list[str]) -> list[dict]:
    return [{"word": t} for t in tokens]


def _align_window(
    tokens, t0, t1, audio, align_model, align_metadata, gpu_lock, sample_rate, depth=0
) -> list[dict]:
    """Align one window; returns one word dict per token (times absolute).

    On CUDA OOM bisects once; on any other failure returns bare (unaligned)
    tokens so neighbors are unaffected.
    """
    import torch
    import whisperx

    duration = len(audio) / sample_rate
    s = max(0.0, t0 - WINDOW_PAD_SEC)
    e = min(duration, t1 + WINDOW_PAD_SEC)
    chunk = audio[int(s * sample_rate): int(e * sample_rate)]
    if len(chunk) < sample_rate // 10:
        return _bare_words(tokens)
    text = " ".join(tokens)
    try:
        with gpu_lock:
            result = whisperx.align(
                [{"text": text, "start": 0.0, "end": len(chunk) / sample_rate}],
                align_model, align_metadata, chunk, "cuda",
                return_char_alignments=False,
            )
        words = [w for seg in result["segments"] for w in seg.get("words", [])]
        if len(words) != len(tokens):
            log.warning("window word-count mismatch (%d vs %d tokens) - marking unaligned",
                        len(words), len(tokens))
            return _bare_words(tokens)
        out = []
        for w in words:
            d = {"word": (w.get("word") or "").strip()}
            if w.get("start") is not None:
                d["start"] = float(w["start"]) + s
            if w.get("end") is not None:
                d["end"] = float(w["end"]) + s
            if w.get("score") is not None:
                d["score"] = w["score"]
            out.append(d)
        return out
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if depth >= 2 or len(tokens) < 2:
            log.error("window OOM at depth %d, giving up on %d tokens", depth, len(tokens))
            return _bare_words(tokens)
        log.warning("window OOM, bisecting (%d tokens, %.0fs)", len(tokens), t1 - t0)
        mid = len(tokens) // 2
        tmid = t0 + (t1 - t0) / 2
        return _align_window(tokens[:mid], t0, tmid, audio, align_model, align_metadata,
                             gpu_lock, sample_rate, depth + 1) + \
               _align_window(tokens[mid:], tmid, t1, audio, align_model, align_metadata,
                             gpu_lock, sample_rate, depth + 1)
    except Exception:
        log.exception("window align failed (%d tokens) - marking unaligned", len(tokens))
        return _bare_words(tokens)


def chunked_align(
    audio,
    transcript_norm: str,
    align_model,
    align_metadata,
    whisper_model,
    gpu_lock,
    batch_size: int,
    sample_rate: int,
    progress_cb=lambda stage, progress=0.0: None,
) -> tuple[list[dict], list[str], float]:
    """Full long-form pipeline. Returns (segments-with-words for postprocess_words,
    warnings, anchor_coverage). Every GT whitespace token appears exactly once."""
    gt_tokens = transcript_norm.split(" ")
    duration = len(audio) / sample_rate
    warnings = []

    progress_cb("transcribing", 0.05)
    with gpu_lock:
        asr = whisper_model.transcribe(audio, batch_size=batch_size)

    progress_cb("anchoring", 0.35)
    asr_tokens, asr_times = asr_token_times(asr["segments"])
    anchors, coverage = find_anchors(gt_tokens, asr_tokens, asr_times)
    log.info("anchoring: %d anchors, coverage %.2f, %d ASR tokens",
             len(anchors), coverage, len(asr_tokens))

    if coverage < MIN_ANCHOR_COVERAGE:
        warnings.append(
            f"transcript does not appear to match audio (anchor coverage "
            f"{coverage:.2f}); returning uniformly distributed unaligned words"
        )
        return ([{"text": transcript_norm, "start": 0.0, "end": duration,
                  "words": _bare_words(gt_tokens)}], warnings, coverage)

    windows = build_windows(gt_tokens, anchors, duration)
    n_low = sum(1 for w in windows if w["low_confidence"])
    if n_low:
        warnings.append(f"{n_low}/{len(windows)} windows had sparse anchors (low confidence)")

    segments = []
    for i, win in enumerate(windows):
        progress_cb(f"aligning window {i + 1}/{len(windows)}",
                    0.4 + 0.55 * i / max(len(windows), 1))
        toks = gt_tokens[win["tok_start"]: win["tok_end"]]
        words = _align_window(toks, win["t0"], win["t1"], audio,
                              align_model, align_metadata, gpu_lock, sample_rate)
        segments.append({"text": " ".join(toks), "start": win["t0"], "end": win["t1"],
                         "words": words})
    progress_cb("postprocessing", 0.97)
    return segments, warnings, coverage
