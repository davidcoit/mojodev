#!/usr/bin/env python3
"""Simulate paired-end spliced RNA-seq reads from a genome FASTA + Ensembl GTF.

Usage: simulate_pairs.py GENOME.fa ANNOTATION.gtf OUT_PREFIX [n_pairs] [read_len] [seed] [frag_mean] [frag_sd]

Writes OUT_PREFIX_1.fastq, OUT_PREFIX_2.fastq and OUT_PREFIX.truth.tsv with one row
per mate: read, mate, chrom, strand, blocks (1-based inclusive genomic blocks of the
error-free mate; several blocks = spans a splice junction).  Mate 1 / mate 2 come from
opposite ends of an RNA fragment (FR library, random fragment orientation).
Run with `python3 -I`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_reads import COMP, mutate, read_fasta, read_transcripts  # noqa: E402


def to_blocks(exons, start, length):
    """Genomic blocks for [start, start+length) of the concatenated exon sequence."""
    blocks, off, need, pos = [], 0, length, start
    for s, e in exons:
        el = e - s + 1
        if pos < off + el and need > 0:
            a = pos - off
            take = min(el - a, need)
            blocks.append((s + a, s + a + take - 1))
            pos += take
            need -= take
        off += el
    return blocks


def main():
    genome_path, gtf_path, prefix = sys.argv[1:4]
    n_pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    read_len = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    frag_mean = float(sys.argv[7]) if len(sys.argv) > 7 else 250.0
    frag_sd = float(sys.argv[8]) if len(sys.argv) > 8 else 50.0
    rng = np.random.default_rng(seed)

    genome = read_fasta(genome_path)
    txs = [t for t in read_transcripts(gtf_path)
           if t[1] in genome and sum(e - s + 1 for s, e in t[3]) >= read_len]
    print(f"{len(txs)} usable transcripts", file=sys.stderr)
    lengths = np.array([sum(e - s + 1 for s, e in t[3]) for t in txs], dtype=float)
    weights = rng.lognormal(0.0, 1.5, len(txs)) * lengths
    weights /= weights.sum()
    picks = rng.choice(len(txs), size=n_pairs, p=weights)
    frag_lens = np.clip(rng.normal(frag_mean, frag_sd, n_pairs), read_len, 1000).astype(int)

    with open(prefix + "_1.fastq", "wb") as f1, open(prefix + "_2.fastq", "wb") as f2, \
            open(prefix + ".truth.tsv", "w") as tr:
        tr.write("read\tmate\tchrom\tstrand\tblocks\n")
        for ri, ti in enumerate(picks):
            tid, chrom, tstrand, exons = txs[ti]
            g = genome[chrom]
            seq = b"".join(g[s - 1:e] for s, e in exons)
            flen = min(int(frag_lens[ri]), len(seq))
            start = int(rng.integers(0, len(seq) - flen + 1))
            frag = seq[start:start + flen]
            # left mate = first read_len of the fragment (forward genome strand),
            # right mate = last read_len, reverse-complemented
            left_blocks = to_blocks(exons, start, read_len)
            right_blocks = to_blocks(exons, start + flen - read_len, read_len)
            left = frag[:read_len]
            right = frag[flen - read_len:].translate(COMP)[::-1]
            if rng.random() < 0.5:   # fragment came off the other strand: mates swap
                m1, m2 = (left, "+", left_blocks), (right, "-", right_blocks)
            else:
                m1, m2 = (right, "-", right_blocks), (left, "+", left_blocks)
            for fq, mate, (rd, strand, blocks) in ((f1, 1, m1), (f2, 2, m2)):
                rd = mutate(rd, rng, 0.005, 0.0005, 0.001)
                fq.write(b"@r%d\n%s\n+\n%s\n" % (ri, rd, b"I" * len(rd)))
                bl = ",".join(f"{a}-{b}" for a, b in blocks)
                tr.write(f"r{ri}\t{mate}\t{chrom}\t{strand}\t{bl}\n")


if __name__ == "__main__":
    main()
