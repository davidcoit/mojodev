#!/usr/bin/env python3
"""Simulate spliced RNA-seq reads from a genome FASTA + Ensembl GTF.

Usage: simulate_reads.py GENOME.fa ANNOTATION.gtf OUT_PREFIX [n_reads] [read_len] [seed]

Writes OUT_PREFIX.fastq and OUT_PREFIX.truth.tsv.  The truth file lists, per read,
the genomic blocks (1-based, inclusive, forward-strand order) the error-free read
came from; more than one block means the read spans a splice junction.
Run with `python3 -I` so nothing in the data directory can shadow stdlib modules.
"""
import re
import sys

import numpy as np

COMP = bytes.maketrans(b"ACGTN", b"TGCAN")
BASES = b"ACGT"


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if name is not None:
                    seqs[name] = b"".join(buf).upper()
                name, buf = line[1:].split()[0].decode(), []
            else:
                buf.append(line.strip())
    seqs[name] = b"".join(buf).upper()
    return seqs


def read_transcripts(path):
    tx = {}
    attr = re.compile(r'transcript_id "([^"]+)"')
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.rstrip("\n").split("\t")
            if c[2] != "exon" or 'gene_biotype "protein_coding"' not in c[8]:
                continue
            tid = attr.search(c[8]).group(1)
            t = tx.setdefault(tid, [c[0], c[6], []])
            t[2].append((int(c[3]), int(c[4])))
    out = []
    for tid, (chrom, strand, exons) in tx.items():
        exons.sort()
        out.append((tid, chrom, strand, exons))
    return out


def mutate(read, rng, sub, indel, nrate):
    """Apply substitutions, small indels, and N calls to a bytes read."""
    r = bytearray(read)
    n = len(r)
    for i in np.nonzero(rng.random(n) < sub)[0]:
        r[i] = BASES[(BASES.index(r[i]) + rng.integers(1, 4)) % 4] if r[i] in BASES else r[i]
    for i in sorted(np.nonzero(rng.random(n) < indel)[0], reverse=True):
        if rng.random() < 0.5:
            del r[i]
        else:
            r.insert(i, BASES[rng.integers(4)])
    for i in np.nonzero(rng.random(len(r)) < nrate)[0]:
        r[i] = ord("N")
    return bytes(r)


def main():
    genome_path, gtf_path, prefix = sys.argv[1:4]
    n_reads = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    read_len = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    rng = np.random.default_rng(seed)

    genome = read_fasta(genome_path)
    txs = [t for t in read_transcripts(gtf_path)
           if t[1] in genome and sum(e - s + 1 for s, e in t[3]) >= read_len]
    print(f"{len(txs)} usable transcripts", file=sys.stderr)

    # Lognormal expression, weighted by length so coverage is roughly even.
    lengths = np.array([sum(e - s + 1 for s, e in t[3]) for t in txs], dtype=float)
    weights = rng.lognormal(0.0, 1.5, len(txs)) * lengths
    weights /= weights.sum()
    picks = rng.choice(len(txs), size=n_reads, p=weights)

    with open(prefix + ".fastq", "wb") as fq, open(prefix + ".truth.tsv", "w") as tr:
        tr.write("read\tchrom\tstrand\tblocks\n")
        for ri, ti in enumerate(picks):
            tid, chrom, tstrand, exons = txs[ti]
            g = genome[chrom]
            # Transcript sequence in forward-genome orientation (exons ascending).
            seq = b"".join(g[s - 1:e] for s, e in exons)
            start = int(rng.integers(0, len(seq) - read_len + 1))
            frag = seq[start:start + read_len]
            # Map [start, start+read_len) in concatenated-exon space to genomic blocks.
            blocks, off, need, pos = [], 0, read_len, start
            for s, e in exons:
                el = e - s + 1
                if pos < off + el and need > 0:
                    a = pos - off
                    take = min(el - a, need)
                    blocks.append((s + a, s + a + take - 1))
                    pos += take
                    need -= take
                off += el
            strand = "+" if rng.random() < 0.5 else "-"
            read = frag if strand == "+" else frag.translate(COMP)[::-1]
            read = mutate(read, rng, 0.005, 0.0005, 0.001)
            qual = b"I" * len(read)
            fq.write(b"@r%d\n%s\n+\n%s\n" % (ri, read, qual))
            bl = ",".join(f"{a}-{b}" for a, b in blocks)
            tr.write(f"r{ri}\t{chrom}\t{strand}\t{bl}\n")


if __name__ == "__main__":
    main()
