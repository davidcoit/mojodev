#!/usr/bin/env python3
"""Paired-end RNA-seq simulator with gene-level truth, run across all cores.

Usage: simulate_rnaseq.py GENOME.fa ANNOTATION.gtf OUT_PREFIX [n_pairs] [read_len] [seed]
                          [nascent_fraction=0.08] [frag_mean=250] [frag_sd=50] [workers]

Differences from simulate_pairs.py:
  * draws from genes of EVERY biotype (piRNA, ncRNA, pseudogene, ...), with lognormal
    gene expression and several isoforms per gene
  * a fraction of fragments are nascent: unspliced, uniform over the gene's genomic span
  * records the true gene of every fragment
  * writes a TRUTH SAM of perfect alignments, so `featureCounts` on it is the oracle count table

Outputs (OUT_PREFIX):
  _1.fastq _2.fastq   reads
  .truth.tsv          read mate chrom strand blocks gene kind(mature|nascent)
  .truth.sam          perfect alignments (name-grouped mates, NH:i:1)
  .origin.tsv         gene_id biotype mature_fragments nascent_fragments
Run with `python3 -I`.
"""
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_reads import COMP, mutate, read_fasta  # noqa: E402
from simulate_pairs import to_blocks  # noqa: E402

G = {}  # shared (copy-on-write after fork) state


def load_annotation(gtf, genome):
    genes, tx = {}, {}
    gid_re = re.compile(r'gene_id "([^"]+)"')
    tid_re = re.compile(r'transcript_id "([^"]+)"')
    bio_re = re.compile(r'gene_biotype "([^"]+)"')
    with open(gtf) as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if c[2] == "gene":
                gid = gid_re.search(c[8]).group(1)
                bio = bio_re.search(c[8])
                genes[gid] = [c[0], c[6], bio.group(1) if bio else "NA", int(c[3]), int(c[4])]
            elif c[2] == "exon":
                tid = tid_re.search(c[8]).group(1)
                t = tx.setdefault(tid, [gid_re.search(c[8]).group(1), c[0], []])
                t[2].append((int(c[3]), int(c[4])))
    by_gene = defaultdict(list)
    for tid, (gid, chrom, ex) in tx.items():
        ex.sort()
        if chrom in genome and gid in genes:
            by_gene[gid].append((tid, ex, sum(e - s + 1 for s, e in ex)))
    return genes, by_gene


def make_chunk(args):
    cid, n, seed = args
    rng = np.random.default_rng([seed, cid])
    S = G
    read_len = S["read_len"]
    genome = S["genome"]
    fq1, fq2, tr, sam = [], [], [], []
    origin = Counter()
    nascent = rng.random(n) < S["nascent_frac"]
    gm = rng.choice(S["n_genes_m"], size=n, p=S["p_mature"])
    gn = rng.choice(S["n_genes_n"], size=n, p=S["p_nascent"])
    flens = np.clip(rng.normal(S["frag_mean"], S["frag_sd"], n), read_len, 1000).astype(int)
    u = rng.random(n)
    swap = rng.random(n) < 0.5
    first = cid * S["chunk"]
    for k in range(n):
        ri = first + k
        flen = int(flens[k])
        if nascent[k]:
            gid = S["names_n"][gn[k]]
            chrom, gstrand, _, gs, ge = S["genes"][gid]
            span = ge - gs + 1
            flen = min(flen, span)
            off = int(u[k] * (span - flen + 1))
            fblocks = [(gs + off, gs + off + flen - 1)]
            lblocks = [(gs + off, gs + off + read_len - 1)]
            rblocks = [(gs + off + flen - read_len, gs + off + flen - 1)]
            kind = "nascent"
        else:
            gid = S["names_m"][gm[k]]
            isoforms = S["isoforms"][gid]
            chrom = S["genes"][gid][0]
            ti = int(np.searchsorted(S["iso_cum"][gid], u[k]))
            ti = min(ti, len(isoforms) - 1)
            tid, exons, tlen = isoforms[ti]
            flen = min(flen, tlen)
            off = int(rng.integers(0, tlen - flen + 1))
            fblocks = to_blocks(exons, off, flen)
            lblocks = to_blocks(exons, off, read_len)
            rblocks = to_blocks(exons, off + flen - read_len, read_len)
            kind = "mature"
        g = genome[chrom]
        frag = b"".join(g[a - 1:b] for a, b in fblocks)
        left = frag[:read_len]
        right = frag[len(frag) - read_len:].translate(COMP)[::-1]
        if swap[k]:
            m1, m2 = (left, "+", lblocks), (right, "-", rblocks)
        else:
            m1, m2 = (right, "-", rblocks), (left, "+", lblocks)
        origin[(gid, kind)] += 1
        lo = min(b[0][0] for b in (lblocks, rblocks))
        hi = max(b[-1][1] for b in (lblocks, rblocks))
        recs = []
        for fq, mate, (rd, strand, blocks) in ((fq1, 1, m1), (fq2, 2, m2)):
            rd2 = mutate(rd, rng, 0.005, 0.0005, 0.001)
            fq.append(b"@r%d\n%s\n+\n%s\n" % (ri, rd2, b"I" * len(rd2)))
            bl = ",".join(f"{a}-{b}" for a, b in blocks)
            tr.append(f"r{ri}\t{mate}\t{chrom}\t{strand}\t{bl}\t{gid}\t{kind}\n")
            cig, prev = [], None
            for a, b in blocks:
                if prev is not None:
                    cig.append(f"{a - prev - 1}N")
                cig.append(f"{b - a + 1}M")
                prev = b
            recs.append((mate, strand, blocks[0][0], "".join(cig)))
        (_, s1, p1, c1), (_, s2, p2, c2) = recs
        tlen_abs = hi - lo + 1
        for (mate, strand, pos, cigar), (omate, ostrand, opos, _) in ((recs[0], recs[1]), (recs[1], recs[0])):
            flag = 1 | 2 | (64 if mate == 1 else 128) | (16 if strand == "-" else 0) | (32 if ostrand == "-" else 0)
            t = tlen_abs if (pos < opos or (pos == opos and mate == 1)) else -tlen_abs
            sam.append(f"r{ri}\t{flag}\t{chrom}\t{pos}\t255\t{cigar}\t=\t{opos}\t{t}\t*\t*\tNH:i:1\n")
    return (b"".join(fq1), b"".join(fq2), "".join(tr), "".join(sam), origin)


def main():
    genome_path, gtf_path, prefix = sys.argv[1:4]
    n_pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    read_len = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    nascent_frac = float(sys.argv[7]) if len(sys.argv) > 7 else 0.08
    frag_mean = float(sys.argv[8]) if len(sys.argv) > 8 else 250.0
    frag_sd = float(sys.argv[9]) if len(sys.argv) > 9 else 50.0
    workers = int(sys.argv[10]) if len(sys.argv) > 10 else os.cpu_count()

    genome = read_fasta(genome_path)
    genes, by_gene = load_annotation(gtf_path, genome)
    rng = np.random.default_rng(seed)

    # genes that can produce a full read: some isoform >= read_len
    names_m, isoforms, iso_cum, avg_len = [], {}, {}, []
    for gid, tl in sorted(by_gene.items()):
        ok = [t for t in tl if t[2] >= read_len]
        if not ok:
            continue
        w = rng.dirichlet(np.ones(len(ok)) * 0.7)
        isoforms[gid] = ok
        iso_cum[gid] = np.cumsum(w)
        names_m.append(gid)
        avg_len.append(float(np.dot(w, [t[2] for t in ok])))
    expr = rng.lognormal(0.0, 2.0, len(names_m))
    wm = expr * np.array(avg_len)
    p_mature = wm / wm.sum()
    # nascent: expression x genomic span, genes spanning at least one read length
    names_n = [g for g in names_m if genes[g][4] - genes[g][3] + 1 >= read_len]
    ex_n = np.array([expr[names_m.index(g)] if False else 0.0 for g in names_n])
    idx = {g: i for i, g in enumerate(names_m)}
    ex_n = np.array([expr[idx[g]] * (genes[g][4] - genes[g][3] + 1) for g in names_n])
    p_nascent = ex_n / ex_n.sum()
    print(f"{len(names_m)} genes can generate reads ({len(by_gene)} annotated with exons); "
          f"{len(names_n)} usable for nascent", file=sys.stderr)

    chunk = 250_000
    G.update(dict(read_len=read_len, genome=genome, genes=genes, isoforms=isoforms, iso_cum=iso_cum,
                  names_m=names_m, names_n=names_n, p_mature=p_mature, p_nascent=p_nascent,
                  n_genes_m=len(names_m), n_genes_n=len(names_n), nascent_frac=nascent_frac,
                  frag_mean=frag_mean, frag_sd=frag_sd, chunk=chunk))
    jobs = [(c, min(chunk, n_pairs - c * chunk), seed) for c in range((n_pairs + chunk - 1) // chunk)]
    origin = Counter()
    with open(prefix + "_1.fastq", "wb") as f1, open(prefix + "_2.fastq", "wb") as f2, \
            open(prefix + ".truth.tsv", "w") as tr, open(prefix + ".truth.sam", "w") as sam:
        tr.write("read\tmate\tchrom\tstrand\tblocks\tgene\tkind\n")
        for ch in sorted(genome):
            sam.write(f"@SQ\tSN:{ch}\tLN:{len(genome[ch])}\n")
        sam.write("@PG\tID:truth\tPN:simulate_rnaseq\n")
        with Pool(workers) as pool:
            for i, (b1, b2, t, s, o) in enumerate(pool.imap(make_chunk, jobs)):
                f1.write(b1)
                f2.write(b2)
                tr.write(t)
                sam.write(s)
                origin.update(o)
                print(f"  chunk {i + 1}/{len(jobs)}", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    with open(prefix + ".origin.tsv", "w") as f:
        f.write("gene_id\tbiotype\tmature\tnascent\n")
        for gid in sorted(genes):
            m, nn = origin.get((gid, "mature"), 0), origin.get((gid, "nascent"), 0)
            f.write(f"{gid}\t{genes[gid][2]}\t{m}\t{nn}\n")


if __name__ == "__main__":
    main()
