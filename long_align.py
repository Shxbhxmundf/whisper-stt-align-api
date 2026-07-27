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
MIN_SCRIPT_RUN_TOKENS = 8  # shorter script switches stay in the surrounding run
RUN_PAD_SEC = 1.5          # run boundaries are anchor-interpolated (~+-2s), pad wide

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
    """Flatten ASR segments to (tokens, time per token).

    Prefers acoustic word-level start times (present when the ASR output was run
    through whisperx.align), interpolating any missing ones from neighbors.
    Falls back to character-midpoint interpolation within the segment span -
    which can drift by seconds inside a ~30s merged segment, so word-level
    times matter for accurate window cuts.
    """
    tokens, times = [], []
    for seg in asr_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        toks = text.split()
        words = seg.get("words") or []
        raw = None
        if len(words) == len(toks):
            raw = [float(w["start"]) if w.get("start") is not None else None for w in words]
            if not any(v is not None for v in raw):
                raw = None
        if raw is None:
            # no word-level info: character-midpoint interpolation
            span = max(end - start, 0.01)
            pos = 0
            for tok in toks:
                mid = pos + len(tok) / 2
                tokens.append(tok)
                times.append(start + span * mid / max(len(text), 1))
                pos += len(tok) + 1
            continue
        known = [i for i, v in enumerate(raw) if v is not None]
        first, last = known[0], known[-1]
        for i in range(len(raw)):
            if raw[i] is not None:
                continue
            if i < first:
                raw[i] = start + (raw[first] - start) * i / max(first, 1)
            elif i > last:
                raw[i] = raw[last] + (end - raw[last]) * (i - last) / max(len(raw) - 1 - last, 1)
            else:
                j0 = max(k for k in known if k < i)
                j1 = min(k for k in known if k > i)
                raw[i] = raw[j0] + (raw[j1] - raw[j0]) * (i - j0) / (j1 - j0)
        tokens.extend(toks)
        times.extend(raw)
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


def _token_script(tok: str) -> str:
    return "hi" if _DEVANAGARI.search(tok) else "en"


def _script_runs(tokens: list[str]) -> list[tuple[int, int, str]]:
    """Split a token span into consecutive same-script runs [(start, end, script)].
    Alignment models are per-script; Latin text fed to the Devanagari model (or
    vice versa) aligns poorly, so mixed-script windows must be aligned per run."""
    runs = []
    for i, tok in enumerate(tokens):
        s = _token_script(tok)
        if runs and runs[-1][2] == s:
            runs[-1] = (runs[-1][0], i + 1, s)
        else:
            runs.append((i, i + 1, s))
    return runs


def _merge_short_runs(runs: list[tuple[int, int, str]],
                      min_tokens: int = MIN_SCRIPT_RUN_TOKENS) -> list[tuple[int, int, str]]:
    """Merge script runs shorter than min_tokens into their larger neighbor.

    Code-switched Hinglish alternates script every few words; per-word model
    switching fragments alignment. Only sustained blocks (a full English
    passage inside a Devanagari script, or vice versa) deserve their own model
    - embedded single words are handled fine by the surrounding run's model.
    """
    runs = list(runs)
    while len(runs) > 1:
        i = min(range(len(runs)), key=lambda k: runs[k][1] - runs[k][0])
        if runs[i][1] - runs[i][0] >= min_tokens:
            break
        left = i - 1 if i > 0 else None
        right = i + 1 if i < len(runs) - 1 else None
        if left is None:
            nb = right
        elif right is None:
            nb = left
        else:
            nb = left if (runs[left][1] - runs[left][0]) >= (runs[right][1] - runs[right][0]) else right
        a, b = sorted((i, nb))
        ra, rb = runs[a], runs[b]
        script = ra[2] if (ra[1] - ra[0]) >= (rb[1] - rb[0]) else rb[2]
        runs[a:b + 1] = [(ra[0], rb[1], script)]
    out = []
    for r in runs:
        if out and out[-1][2] == r[2]:
            out[-1] = (out[-1][0], r[1], r[2])
        else:
            out.append(r)
    return out


def _anchor_time_fn(anchors: list[tuple[int, float]], n_tokens: int, duration: float):
    """Piecewise-linear token-index -> audio-time map through the anchors."""
    xs = [0] + [a[0] for a in anchors if 0 < a[0] < n_tokens] + [n_tokens]
    ys = [0.0] + [a[1] for a in anchors if 0 < a[0] < n_tokens] + [duration]

    def f(i: float) -> float:
        j = min(max(bisect.bisect_right(xs, i) - 1, 0), len(xs) - 2)
        x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (i - x0) / (x1 - x0)

    return f


def _align_window(
    tokens, t0, t1, audio, align_model, align_metadata, gpu_lock, sample_rate,
    depth=0, pad=WINDOW_PAD_SEC,
) -> list[dict]:
    """Align one window; returns one word dict per token (times absolute).

    On CUDA OOM bisects once; on any other failure returns bare (unaligned)
    tokens so neighbors are unaffected.
    """
    import torch
    import whisperx

    duration = len(audio) / sample_rate
    s = max(0.0, t0 - pad)
    e = min(duration, t1 + pad)
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
                             gpu_lock, sample_rate, depth + 1, pad) + \
               _align_window(tokens[mid:], tmid, t1, audio, align_model, align_metadata,
                             gpu_lock, sample_rate, depth + 1, pad)
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
    get_align_model_fn=None,
) -> tuple[list[dict], list[str], float, dict]:
    """Full long-form pipeline. Returns (segments-with-words for postprocess_words,
    warnings, anchor_coverage, stats). Every GT whitespace token appears exactly once."""
    import whisperx

    gt_tokens = transcript_norm.split(" ")
    duration = len(audio) / sample_rate
    warnings = []

    stats = {"asr_word_timing": False, "asr_language": None}
    progress_cb("transcribing", 0.05)
    with gpu_lock:
        asr = whisper_model.transcribe(audio, batch_size=batch_size)
    asr_segments = asr["segments"]
    stats["asr_language"] = asr.get("language")

    # Acoustic word times for the ASR stream (align model picked by the ASR
    # text's own script - Whisper may emit Devanagari against a Latin GT).
    # Without this, anchor times come from char interpolation inside ~30s
    # merged segments and drift by seconds, cutting windows in the wrong place.
    if get_align_model_fn is not None and asr_segments:
        asr_text = " ".join(s.get("text", "") for s in asr_segments)
        n_dev = len(_DEVANAGARI.findall(asr_text))
        n_lat = sum(1 for c in asr_text if c.isascii() and c.isalpha())
        am2 = get_align_model_fn("hi" if n_dev > n_lat else "en")
        if am2 is not None:
            progress_cb("timing asr words", 0.25)
            try:
                with gpu_lock:
                    asr_segments = whisperx.align(
                        asr_segments, am2[0], am2[1], audio, "cuda",
                        return_char_alignments=False,
                    )["segments"]
                stats["asr_word_timing"] = True
            except Exception:
                log.exception("asr word-timing pass failed; using segment interpolation")

    progress_cb("anchoring", 0.35)
    asr_tokens, asr_times = asr_token_times(asr_segments)
    anchors, coverage = find_anchors(gt_tokens, asr_tokens, asr_times)
    log.info("anchoring: %d anchors, coverage %.2f, %d ASR tokens",
             len(anchors), coverage, len(asr_tokens))

    if coverage < MIN_ANCHOR_COVERAGE:
        warnings.append(
            f"transcript does not appear to match audio (anchor coverage "
            f"{coverage:.2f}); returning uniformly distributed unaligned words"
        )
        return ([{"text": transcript_norm, "start": 0.0, "end": duration,
                  "words": _bare_words(gt_tokens)}], warnings, coverage, stats)

    windows = build_windows(gt_tokens, anchors, duration)
    n_low = sum(1 for w in windows if w["low_confidence"])
    if n_low:
        warnings.append(f"{n_low}/{len(windows)} windows had sparse anchors (low confidence)")

    # Each same-script run inside a window is aligned with the model matching
    # ITS script (run boundaries timed via the anchor map). Feeding Latin text
    # to the Devanagari model (or vice versa) smears the alignment.
    tok_time = _anchor_time_fn(anchors, len(gt_tokens), duration)
    models = {}
    if get_align_model_fn is not None:
        for lang in ("en", "hi"):
            models[lang] = get_align_model_fn(lang)
    default_model = (align_model, align_metadata)

    segments, n_runs = [], 0
    for i, win in enumerate(windows):
        progress_cb(f"aligning window {i + 1}/{len(windows)}",
                    0.4 + 0.55 * i / max(len(windows), 1))
        toks = gt_tokens[win["tok_start"]: win["tok_end"]]
        for r0, r1, script in _merge_short_runs(_script_runs(toks)):
            n_runs += 1
            run_toks = toks[r0:r1]
            g0, g1 = win["tok_start"] + r0, win["tok_start"] + r1
            full_window = r0 == 0 and r1 == len(toks)
            t0 = win["t0"] if r0 == 0 else max(win["t0"], tok_time(g0))
            t1 = win["t1"] if r1 == len(toks) else min(win["t1"], tok_time(g1))
            t1 = max(t1, t0 + 0.05)
            am_run = models.get(script) or default_model
            words = _align_window(
                run_toks, t0, t1, audio, am_run[0], am_run[1], gpu_lock, sample_rate,
                pad=WINDOW_PAD_SEC if full_window else RUN_PAD_SEC,
            )
            segments.append({"text": " ".join(run_toks), "start": t0, "end": t1,
                             "words": words})
    stats.update({"n_windows": len(windows), "n_runs": n_runs,
                  "n_anchors": len(anchors)})
    progress_cb("postprocessing", 0.97)
    return segments, warnings, coverage, stats
