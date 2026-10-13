#!/usr/bin/env python3
"""Count questions in PDF question banks / exam papers. Fully local (pdfplumber).

Usage:
    python count_questions.py <file.pdf | folder>
"""
import argparse
import re
import sys
from pathlib import Path

import pdfplumber

MIN_TEXT_CHARS = 30
MIN_RUN_FOR_WEAK = 5  # roman / alphabet / circled need a run this long to count

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"

PATTERNS = {
    "num_dot": re.compile(r"^(\d{1,3})\.\s+(?=\S)", re.M),
    "num_paren": re.compile(r"^(\d{1,3})\)\s+", re.M),
    "num_bare": re.compile(r"^(\d{1,3})\s+(?=[A-Z])", re.M),
    "q_prefix": re.compile(r"^Q(\d{1,3})\.", re.M),
    "problem_n": re.compile(r"^Problem\s+(\d{1,3})\b", re.M),
    "answer_marker": re.compile(r"^Answer:\s*[A-Za-z]", re.M),
    "roman": re.compile(r"^([ivxlcdm]{1,6})[\.\)]\s+(?=[A-Za-z])", re.M | re.I),
    "alphabet": re.compile(r"^([a-zA-Z])[\.\)]\s+(?=[A-Za-z])", re.M),
    "circled": re.compile(r"^([" + CIRCLED + r"])\s*", re.M),
}
WEAK = {"roman", "alphabet", "circled"}

ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def roman_to_int(s):
    s = s.lower()
    total = 0
    for i, ch in enumerate(s):
        v = ROMAN_VALUES[ch]
        total += -v if i + 1 < len(s) and ROMAN_VALUES[s[i + 1]] > v else v
    return total


def to_numbers(name, matches):
    """Convert captured tokens to ints so sequential runs can be measured."""
    if name == "roman":
        return [roman_to_int(m) for m in matches]
    if name == "alphabet":
        return [ord(m.lower()) - ord("a") + 1 for m in matches]
    if name == "circled":
        return [CIRCLED.index(m) + 1 for m in matches]
    if name in ("answer_marker",):
        return list(range(1, len(matches) + 1))  # no numbers; treat as sequential
    return [int(m) for m in matches]


def split_runs(nums):
    """A new run starts whenever the next number isn't exactly last + 1."""
    runs = []
    for n in nums:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


def filter_bare(text):
    """Keep a num_bare match only if it plugs a gap in the num_dot sequence.

    Walk dotted and bare matches in document order, tracking the running max
    of confirmed numbers; accept a bare number only if it equals max + 1, and
    then treat it as confirmed so the next bare number can chain from it.
    """
    events = [(m.start(), int(m.group(1)), False)
              for m in PATTERNS["num_dot"].finditer(text)]
    events += [(m.start(), int(m.group(1)), True)
               for m in PATTERNS["num_bare"].finditer(text)]
    events.sort()
    running, dot_pos, accepted_pos = 0, [], []
    for pos, n, is_bare in events:
        if is_bare:
            if n == running + 1:
                accepted_pos.append(pos)
                running = n
        else:
            dot_pos.append(pos)
            running = n if n > running else running
    return dot_pos, accepted_pos


def longest_run_span(nums, positions):
    """Return (length, (start, end)) of the longest sequential run."""
    best, best_span, i = 0, None, 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i + 1 > best:
            best, best_span = j - i + 1, (positions[i], positions[j])
        i = j + 1
    return best, best_span


def merge_clusters(buckets):
    """buckets: list of (start, end, count). Overlapping spans compete (max);
    disjoint clusters are separate sections (sum)."""
    total, cur_end, cur_max = 0, None, 0
    for start, end, count in sorted(buckets):
        if cur_end is not None and start <= cur_end:
            cur_end = max(cur_end, end)
            cur_max = max(cur_max, count)
        else:
            total += cur_max
            cur_end, cur_max = end, count
    return total + cur_max


def extract_text(path):
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def analyse(path):
    res = {"filename": path.name, "total": 0, "fired": [], "runs": 0,
           "flag": "clean", "notes": []}
    try:
        text = extract_text(path)
    except Exception as e:  # corrupt / encrypted PDFs
        res.update(flag="needs review", notes=[f"could not read PDF: {e}"])
        return res

    if len(text.strip()) < MIN_TEXT_CHARS:
        res.update(flag="needs review", notes=["needs OCR / manual review"])
        res["ocr"] = True
        return res

    counts, run_info, spans = {}, {}, {}
    dot_pos, bare_pos = filter_bare(text)
    for name, rx in PATTERNS.items():
        if name == "num_bare":
            counts[name] = len(bare_pos)
            continue
        ms = list(rx.finditer(text))
        matches = [m.group(1) if rx.groups else m.group(0) for m in ms]
        if not matches:
            counts[name] = 0
            continue
        positions = [m.start() for m in ms]
        nums = to_numbers(name, matches)
        runs = split_runs(nums)
        if name in WEAK:
            best, span = longest_run_span(nums, positions)
            if best < MIN_RUN_FOR_WEAK:
                res["notes"].append(f"{name}: {len(matches)} stray hit(s) ignored")
                counts[name] = 0
                continue
            counts[name] = best
            spans[name] = span
        else:
            counts[name] = len(matches)
            spans[name] = (positions[0], positions[-1])
        run_info[name] = len(runs)

    # merged "numbered" bucket: num_dot plus bare numbers that plug gaps
    numbered_pos = sorted(dot_pos + bare_pos)
    if numbered_pos:
        counts["num_dot"] = len(numbered_pos) - len(bare_pos)
        spans["numbered"] = (numbered_pos[0], numbered_pos[-1])

    res["fired"] = [n for n, c in counts.items() if c]
    buckets = []
    if numbered_pos:
        buckets.append((*spans["numbered"], len(numbered_pos)))
    for name in ("num_paren", "answer_marker", "roman", "alphabet", "circled"):
        if counts[name]:
            buckets.append((*spans[name], counts[name]))
    res["total"] = (merge_clusters(buckets) + counts["q_prefix"]
                    + counts["problem_n"])

    # run count: from the pattern(s) that drive the total
    drivers = [n for n in res["fired"]]
    res["runs"] = sum(run_info[n] for n in drivers
                      if n in ("q_prefix", "problem_n")) + max(
        [run_info.get(n, 0) for n in res["fired"]
         if n not in ("q_prefix", "problem_n")] or [0])

    if res["total"] == 0:
        res["flag"] = "needs review"
        res["notes"].append("zero matches")
    elif res["runs"] > 6 or any(n in WEAK for n in res["fired"]):
        res["flag"] = "check"
        if res["runs"] > 6:
            res["notes"].append(f"{res['runs']} runs (many restarts / gaps)")
    elif res["runs"] > 1:
        res["notes"].append(f"{res['runs']} runs (likely per-section numbering)")
    return res


def collect_pdfs(target):
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() == ".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="PDF file or folder of PDFs")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        sys.exit(f"Path not found: {target}")
    pdfs = collect_pdfs(target)
    if not pdfs:
        sys.exit("No PDF files found.")

    results = [analyse(p) for p in pdfs]

    if target.is_file():
        r = results[0]
        if r.get("ocr"):
            print(f"{r['filename']}: needs OCR / manual review")
        else:
            print(f"Total number of questions in {r['filename']}: {r['total']}")
    else:
        for r in results:
            print(f"{r['filename']}: " + ("needs OCR / manual review" if r.get("ocr")
                                          else str(r["total"])))
        flagged = [r for r in results if r.get("ocr") or r["total"] == 0]
        if flagged:
            print("\nNotes:")
            for r in flagged:
                why = "needs OCR / manual review" if r.get("ocr") else "zero matches"
                print(f"  - {r['filename']}: {why}")


if __name__ == "__main__":
    main()
