#!/usr/bin/env python3
"""Infer library strandedness from a paired-end SAM and a GTF (like RSeQC infer_experiment).

Usage: infer_strand.py ALIGNED.sam ANNOTATION.gtf [max_records]
Uses unique (MAPQ >= 10), properly paired mate-1 alignments that overlap exactly one
gene.  Reports the fraction where mate 1 is on the gene's strand:
  ~0.5  unstranded                      -> featureCounts -s 0
  ~1.0  mate 1 sense (ligation-style)   -> -s 1
  ~0.0  mate 1 antisense (dUTP / TruSeq)-> -s 2
Run with `python3 -I`.
"""
import bisect
import re
import sys
from collections import defaultdict


def load_genes(gtf):
    genes = {}
    with open(gtf) as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if c[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', c[8]).group(1)
            genes[gid] = (c[0], int(c[3]), int(c[4]), c[6])
    by_chrom = defaultdict(list)
    for gid, (ch, s, e, st) in genes.items():
        by_chrom[ch].append((s, e, st))
    for ch in by_chrom:
        by_chrom[ch].sort()
    return by_chrom


def main():
    sam, gtf = sys.argv[1:3]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    genes = load_genes(gtf)
    starts = {ch: [g[0] for g in gl] for ch, gl in genes.items()}
    maxlen = {ch: max(g[1] - g[0] for g in gl) for ch, gl in genes.items()}
    same = opp = skipped = 0
    n = 0
    with open(sam) as f:
        for line in f:
            if line[0] == "@":
                continue
            c = line.split("\t", 9)
            flag = int(c[1])
            if flag & 4 or not flag & 64 or not flag & 2 or int(c[4]) < 10:
                continue
            n += 1
            if n > limit:
                break
            ch, pos = c[2], int(c[3])
            gl = genes.get(ch)
            if gl is None:
                continue
            hit = []
            i = bisect.bisect_right(starts[ch], pos)
            j = i - 1
            while j >= 0 and gl[j][0] >= pos - maxlen[ch]:
                if gl[j][0] <= pos <= gl[j][1]:
                    hit.append(gl[j])
                j -= 1
            if len(hit) != 1:
                skipped += 1
                continue
            strand = "-" if flag & 16 else "+"
            if strand == hit[0][2]:
                same += 1
            else:
                opp += 1
    tot = same + opp
    print(f"mate-1 records used: {tot} (skipped {skipped} overlapping/intergenic)")
    print(f"mate 1 on the gene strand: {same} ({100*same/max(tot,1):.1f}%) | opposite: {opp} ({100*opp/max(tot,1):.1f}%)")


if __name__ == "__main__":
    main()
