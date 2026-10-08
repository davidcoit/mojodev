#!/usr/bin/env python3
"""Score a SAM file against simulate_reads.py truth.

Usage: score_sam.py ALIGNED.sam TRUTH.tsv

Reports, split by whether the read truly spans a splice junction:
  mapped      read has an alignment
  locus       alignment overlaps a true exon block on the right chrom/strand
  blocks      aligned reference blocks equal the true blocks exactly
  junctions   the set of introns (start,end) equals the true set
  jn_off<=2   same number of junctions, each endpoint within 2 bp of truth
Run with `python3 -I`.
"""
import re
import sys
from collections import Counter

CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def sam_blocks(pos, cigar):
    """1-based inclusive reference blocks from POS + CIGAR (D merges, N splits)."""
    blocks, cur = [], pos
    start = pos
    for n, op in CIG.findall(cigar):
        n = int(n)
        if op in "M=XD":
            cur += n
        elif op == "N":
            blocks.append((start, cur - 1))
            cur += n
            start = cur
    blocks.append((start, cur - 1))
    return blocks


def introns(blocks):
    return [(blocks[i][1] + 1, blocks[i + 1][0] - 1) for i in range(len(blocks) - 1)]


def main():
    sam_path, truth_path = sys.argv[1:3]
    truth = {}
    with open(truth_path) as f:
        next(f)
        for line in f:
            r, chrom, strand, bl = line.rstrip("\n").split("\t")
            blocks = [tuple(map(int, b.split("-"))) for b in bl.split(",")]
            truth[r] = (chrom, strand, blocks)

    stats = {k: Counter() for k in ("spliced", "unspliced", "all")}
    mapq_wrong = Counter()
    mapq_total = Counter()
    with open(sam_path) as f:
        for line in f:
            if line[0] == "@":
                continue
            c = line.rstrip("\n").split("\t")
            name, flag, chrom, pos, mapq, cigar = c[0], int(c[1]), c[2], int(c[3]), int(c[4]), c[5]
            tchrom, tstrand, tblocks = truth[name]
            kind = "spliced" if len(tblocks) > 1 else "unspliced"
            for k in (kind, "all"):
                stats[k]["n"] += 1
            if flag & 4:
                continue
            blocks = sam_blocks(pos, cigar)
            strand = "-" if flag & 16 else "+"
            ok_locus = (chrom == tchrom and strand == tstrand and
                        any(a <= tb and b >= ta for ta, tb in [(x, y) for x, y in tblocks] for a, b in blocks))
            ok_blocks = chrom == tchrom and strand == tstrand and blocks == tblocks
            ti, ai = introns(tblocks), introns(blocks)
            ok_jn = ok_locus and ti == ai
            ok_near = (ok_locus and len(ti) == len(ai) and
                       all(abs(a[0] - t[0]) <= 2 and abs(a[1] - t[1]) <= 2 for a, t in zip(ai, ti)))
            mq = "mq0-9" if mapq < 10 else "mq10+"
            mapq_total[mq] += 1
            if not ok_locus:
                mapq_wrong[mq] += 1
            for k in (kind, "all"):
                s = stats[k]
                s["mapped"] += 1
                s["locus"] += ok_locus
                s["blocks"] += ok_blocks
                s["junctions"] += ok_jn
                s["jn_off<=2"] += ok_near

    print(f"{'':10s} {'reads':>8s} {'mapped':>8s} {'locus':>8s} {'blocks':>8s} {'junctions':>9s} {'jn<=2bp':>8s}")
    for k in ("unspliced", "spliced", "all"):
        s = stats[k]
        n = max(s["n"], 1)
        print(f"{k:10s} {s['n']:8d} {100*s['mapped']/n:7.2f}% {100*s['locus']/n:7.2f}% "
              f"{100*s['blocks']/n:7.2f}% {100*s['junctions']/n:8.2f}% {100*s['jn_off<=2']/n:7.2f}%")
    for mq in ("mq0-9", "mq10+"):
        t = max(mapq_total[mq], 1)
        print(f"{mq}: {mapq_total[mq]} mapped, {mapq_wrong[mq]} wrong locus ({100*mapq_wrong[mq]/t:.3f}%)")


if __name__ == "__main__":
    main()
