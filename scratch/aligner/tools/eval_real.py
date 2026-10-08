#!/usr/bin/env python3
"""Truth-free evaluation of a (paired-end) SAM on real RNA-seq.

Usage: eval_real.py ALIGNED.sam GENOME.fa ANNOTATION.gtf [OTHER.sam]

Reports mapping / proper-pair rates, how many distinct introns are annotated vs novel,
the splice-motif composition of novel introns (real introns are overwhelmingly
GT..AG / GC..AG / AT..AC), the insert-size distribution of unspliced proper pairs, and,
if OTHER.sam is given, per-mate agreement with it (same chrom/strand/start).
Run with `python3 -I`.
"""
import re
import sys
from collections import Counter

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from score_sam import sam_blocks, introns  # noqa: E402
from simulate_reads import read_fasta  # noqa: E402

CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def annotated_introns(gtf):
    tx = {}
    with open(gtf) as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if c[2] != "exon":
                continue
            tid = re.search(r'transcript_id "([^"]+)"', c[8]).group(1)
            tx.setdefault(tid, [c[0], []])[1].append((int(c[3]), int(c[4])))
    out = set()
    for chrom, ex in tx.values():
        ex.sort()
        for i in range(len(ex) - 1):
            out.add((chrom, ex[i][1] + 1, ex[i + 1][0] - 1))
    return out


def load_sam(path, want_pairs=True):
    recs = []
    seen = set()
    with open(path) as f:
        for line in f:
            if line[0] == "@":
                continue
            c = line.rstrip("\n").split("\t")
            flag = int(c[1])
            if flag & 256 or flag & 2048:  # secondary / supplementary: evaluate primary records only
                continue
            if not flag & 192 and not flag & 1:
                # single-end style records (e.g. minimap2 splice:sr given two files):
                # the first record of a name is mate 1, the second is mate 2
                flag |= 128 if c[0] in seen else 64
                seen.add(c[0])
            recs.append((c[0], flag, c[2], int(c[3]), int(c[4]), c[5], int(c[8])))
    return recs


def main():
    sam, genome_path, gtf = sys.argv[1:4]
    other = sys.argv[4] if len(sys.argv) > 4 else None
    ann = annotated_introns(gtf)
    genome = read_fasta(genome_path)
    recs = load_sam(sam)

    n = len(recs)
    mapped = sum(1 for r in recs if not r[1] & 4)
    paired = sum(1 for r in recs if r[1] & 1)
    proper = sum(1 for r in recs if r[1] & 2)
    mq10 = sum(1 for r in recs if not r[1] & 4 and r[4] >= 10)
    print(f"records {n} | mapped {100*mapped/n:.2f}% | MAPQ>=10 {100*mq10/n:.2f}%"
          + (f" | proper-pair flag {100*proper/n:.2f}%" if paired else ""))

    intron_counts = Counter()
    spliced = 0
    for name, flag, chrom, pos, mapq, cigar, tlen in recs:
        if flag & 4 or "N" not in cigar:
            continue
        spliced += 1
        for s, e in introns(sam_blocks(pos, cigar)):
            intron_counts[(chrom, s, e)] += 1
    ann_hits = sum(1 for k in intron_counts if k in ann)
    ann_reads = sum(v for k, v in intron_counts.items() if k in ann)
    tot_reads = sum(intron_counts.values())
    print(f"spliced reads {spliced} ({100*spliced/max(mapped,1):.1f}% of mapped) | distinct introns {len(intron_counts)} "
          f"| annotated {ann_hits} ({100*ann_hits/max(len(intron_counts),1):.1f}%) "
          f"| reads on annotated introns {100*ann_reads/max(tot_reads,1):.2f}% of intron observations")

    motifs = Counter()
    novel_motifs = Counter()
    for (chrom, s, e), v in intron_counts.items():
        g = genome.get(chrom)
        if g is None or e - s < 3:
            continue
        m = (g[s - 1:s + 1] + b".." + g[e - 2:e]).decode()
        d, a = m[:2], m[-2:]
        key = d + "-" + a
        rc = {"GT-AG": "GT-AG", "CT-AC": "GT-AG(-)", "GC-AG": "GC-AG", "CT-GC": "GC-AG(-)",
              "AT-AC": "AT-AC", "GT-AT": "AT-AC(-)"}.get(key, "other")
        motifs[rc] += 1
        if (chrom, s, e) not in ann:
            novel_motifs[rc] += 1
    tm = max(sum(motifs.values()), 1)
    tn = max(sum(novel_motifs.values()), 1)
    print("motifs, all introns:   " + ", ".join(f"{k} {100*v/tm:.1f}%" for k, v in motifs.most_common()))
    print("motifs, novel introns: " + ", ".join(f"{k} {100*v/tn:.1f}%" for k, v in novel_motifs.most_common()))

    sizes = [abs(r[6]) for r in recs if r[1] & 2 and r[1] & 64 and "N" not in r[5] and r[6] != 0]
    if sizes:
        sizes.sort()
        q = lambda p: sizes[int(p * (len(sizes) - 1))]
        print(f"insert size (unspliced proper pairs, n={len(sizes)}): median {q(.5)}, IQR {q(.25)}-{q(.75)}, 5-95% {q(.05)}-{q(.95)}")

    if other:
        mine = {(r[0], r[1] & 192): (r[2], r[3], r[1] & 16, r[1] & 4) for r in recs}
        both = same = only_me = only_them = 0
        for r in load_sam(other):
            if r[1] & 256 or r[1] & 2048:
                continue
            k = (r[0], r[1] & 192)
            m = mine.get(k)
            if m is None:
                continue
            theirs_mapped = not r[1] & 4
            mine_mapped = not m[3]
            if mine_mapped and theirs_mapped:
                both += 1
                same += (m[0], m[2]) == (r[2], r[1] & 16) and abs(m[1] - r[3]) <= 5
            elif mine_mapped:
                only_me += 1
            elif theirs_mapped:
                only_them += 1
        print(f"vs other SAM: both mapped {both} | same locus (+-5bp start) {100*same/max(both,1):.2f}% "
              f"| only mine mapped {only_me} | only other mapped {only_them}")


if __name__ == "__main__":
    main()
