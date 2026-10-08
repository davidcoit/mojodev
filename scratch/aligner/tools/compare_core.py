#!/usr/bin/env python3
"""Per-fragment comparison of two arms' featureCounts -R CORE assignments (no truth needed).

Usage: compare_core.py CMPDIR ARM_A ARM_B
Cross-tabulates what ARM_A and ARM_B did with every fragment: assigned to the same / a different
gene, one of them multi-mapping / ambiguous / no feature, or the fragment absent from one arm
(e.g. STAR omits reads it could not align).  Run with `python3 -I`.
"""
import os
import sys
from collections import Counter


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            d[c[0]] = (c[1], c[3] if len(c) > 3 else "")
    return d


def main():
    root, a, b = sys.argv[1:4]
    B = load(os.path.join(root, b, "aligned.sam.featureCounts"))
    cats = Counter()
    seen = set()
    with open(os.path.join(root, a, "aligned.sam.featureCounts")) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            name, st, tg = c[0], c[1], (c[3] if len(c) > 3 else "")
            seen.add(name)
            o = B.get(name)
            if o is None:
                cats[f"{a} {st}; absent from {b}"] += 1
            elif st == "Assigned" and o[0] == "Assigned":
                cats["both assigned, same gene" if tg == o[1] else "both assigned, DIFFERENT gene"] += 1
            else:
                cats[f"{a} {st}; {b} {o[0]}"] += 1
    for name, o in B.items():
        if name not in seen:
            cats[f"absent from {a}; {b} {o[0]}"] += 1
    tot = sum(cats.values())
    print(f"{tot} distinct fragments across {a} and {b}\n")
    print("| outcome | fragments | share |\n|---|---:|---:|")
    for k, v in cats.most_common(14):
        print(f"| {k} | {v} | {100 * v / tot:.3f}% |")


if __name__ == "__main__":
    main()
