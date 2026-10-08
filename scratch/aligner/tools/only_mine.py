#!/usr/bin/env python3
"""Characterize mates mapped by SAM A but unmapped in SAM B.

Usage: only_mine.py A.sam B.sam
B may be single-end style (mate inferred from record order). Prints the flag, MAPQ,
alignment-score, clipping and proper-pair profile of the A-only mates vs all A mates.
"""
import re
import sys
from collections import Counter

CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def mates(path):
    seen = set()
    with open(path) as f:
        for line in f:
            if line[0] == "@":
                continue
            c = line.split("\t")
            flag = int(c[1])
            if not flag & 192 and not flag & 1:
                flag |= 128 if c[0] in seen else 64
                seen.add(c[0])
            yield c, flag & 192, flag


def main():
    a_path, b_path = sys.argv[1:3]
    b_unmapped = set()
    for c, m, flag in mates(b_path):
        if flag & 4:
            b_unmapped.add((c[0], m))
    prof = {"only": Counter(), "all": Counter()}
    for c, m, flag in mates(a_path):
        if flag & 4:
            continue
        key = "only" if (c[0], m) in b_unmapped else "all"
        for k in {key, "all"}:
            p = prof[k]
            p["n"] += 1
            mapq = int(c[4])
            p["mapq>=10"] += mapq >= 10
            p["proper"] += bool(flag & 2)
            clip = sum(int(n) for n, op in CIG.findall(c[5]) if op == "S")
            p["clip>=20"] += clip >= 20
            p["clip>=40"] += clip >= 40
            p["spliced"] += "N" in c[5]
            mm = re.search(r"NM:i:(\d+)", line_tags(c))
            p["nm_sum"] += int(mm.group(1)) if mm else 0
            asm = re.search(r"AS:i:(\d+)", line_tags(c))
            p["as_sum"] += int(asm.group(1)) if asm else 0
    for k in ("only", "all"):
        p = prof[k]
        n = max(p["n"], 1)
        print(f"{k:5s} n={p['n']:9d} MAPQ>=10 {100*p['mapq>=10']/n:5.1f}% | proper {100*p['proper']/n:5.1f}% | "
              f"clip>=20 {100*p['clip>=20']/n:5.1f}% clip>=40 {100*p['clip>=40']/n:5.1f}% | spliced {100*p['spliced']/n:5.1f}% | "
              f"mean NM {p['nm_sum']/n:.2f} | mean AS {p['as_sum']/n:.1f}")


def line_tags(c):
    return "\t".join(c[11:])


if __name__ == "__main__":
    main()
