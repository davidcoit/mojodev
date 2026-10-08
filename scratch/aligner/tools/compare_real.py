#!/usr/bin/env python3
"""Report for a real-data run_compare.sh directory (no ground truth).

Usage: compare_real.py CMPDIR GTF GENOME [REFERENCE_ARM=star_annot] [OUT.md]

Reference-based: every other arm is compared with REFERENCE_ARM (STAR with the annotated index),
which is a reference point and NOT the truth.  Sections: timings, truth-free alignment profile
(tools/eval_real.py), featureCounts outcome shares, gene-count agreement with the reference,
and the most discordant genes with their names / biotypes.  Run with `python3 -I`.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_report import SUMMARY_KEYS, VARIANTS, count_metrics, read_counts, read_summary  # noqa: E402


def gene_info(gtf):
    info = {}
    with open(gtf) as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if c[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', c[8]).group(1)
            nm = re.search(r'gene_name "([^"]+)"', c[8])
            bio = re.search(r'gene_biotype "([^"]+)"', c[8])
            info[gid] = (nm.group(1) if nm else "-", bio.group(1) if bio else "-", f"{c[0]}:{c[3]}-{c[4]}({c[6]})")
    return info


def main():
    cmp_dir, gtf, genome = sys.argv[1:4]
    ref_arm = sys.argv[4] if len(sys.argv) > 4 else "star_annot"
    out_md = sys.argv[5] if len(sys.argv) > 5 else os.path.join(cmp_dir, "report.md")
    arms = [d for d in ["gpu", "star_annot", "star_annot40k", "star_denovo2p"] if os.path.isdir(os.path.join(cmp_dir, d))]
    info = gene_info(gtf)
    L = ["## Timings\n", "| step | wall (s) | CPU (s) | peak RSS (GB) |\n|---|---:|---:|---:|"]
    tj = os.path.join(cmp_dir, "timings.jsonl")
    if os.path.exists(tj):
        for line in open(tj):
            r = json.loads(line)
            L.append(f"| {r['label'].split('.', 1)[1]} | {r['wall_s']} | {r['cpu_s']} | {r['maxrss_gb']} |")

    L.append("\n## Alignment profile (no truth)\n")
    for a in arms:
        sam = os.path.join(cmp_dir, a, "aligned.sam")
        res = subprocess.run(["python3", "-I", os.path.join(HERE, "eval_real.py"), sam, genome, gtf],
                             capture_output=True, text=True)
        L.append(f"**{a}**\n\n```\n{res.stdout.strip()}\n```\n")

    L.append("\n## featureCounts outcome (share of fragments)\n")
    for v in VARIANTS:
        L.append(f"\n**variant `{v}`**\n")
        L.append("| arm | Assigned | NoFeatures | Ambiguity | MultiMapping | Unmapped | Singleton | fragments |\n|---|---:|---:|---:|---:|---:|---:|---:|")
        for a in arms:
            p = os.path.join(cmp_dir, a, f"counts_{v}.txt.summary")
            if not os.path.exists(p):
                continue
            s = read_summary(p)
            tot = sum(s.values())
            cells = " | ".join(f"{100 * s.get(k, 0) / tot:.2f}%" for k in SUMMARY_KEYS[:4] + SUMMARY_KEYS[4:6])
            L.append(f"| {a} | {cells} | {int(tot)} |")

    L.append(f"\n## Gene-count agreement with `{ref_arm}` (a reference, not the truth)\n")
    for v in VARIANTS:
        rp = os.path.join(cmp_dir, ref_arm, f"counts_{v}.txt")
        if not os.path.exists(rp):
            continue
        ids, ref = read_counts(rp)
        L.append(f"\n**variant `{v}`**\n")
        L.append("| arm | genes | Pearson (log2) | Spearman | RMSE log2 | genes ref>=10 | median rel diff | within 10% | off by >2x | L1 diff | total vs ref |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a in arms:
            if a == ref_arm:
                continue
            p = os.path.join(cmp_dir, a, f"counts_{v}.txt")
            if not os.path.exists(p):
                continue
            ids2, cnt = read_counts(p)
            m = count_metrics(cnt, ref)
            L.append(f"| {a} | {m['genes_counted']} | {m['pearson_log2']:.4f} | {m['spearman']:.4f} | {m['rmse_log2']:.3f} | "
                     f"{m['genes_ge10']} | {100 * m['median_rel_err']:.2f}% | {100 * m['within_10pct']:.1f}% | "
                     f"{100 * m['off_by_2x']:.2f}% | {100 * m['L1_frac']:.2f}% | {m['total']:.0f} / {m['oracle_total']:.0f} |")

    # most discordant genes, GPU vs reference, default policy
    rp = os.path.join(cmp_dir, ref_arm, "counts_all.txt")
    gp = os.path.join(cmp_dir, "gpu", "counts_all.txt")
    if os.path.exists(rp) and os.path.exists(gp):
        ids, ref = read_counts(rp)
        _, gpu = read_counts(gp)
        keep = np.where((ref + gpu) >= 200)[0]
        ratio = np.log2((gpu[keep] + 1) / (ref[keep] + 1))
        order = np.argsort(-np.abs(ratio))[:25]
        L.append(f"\n## Most discordant genes, `gpu` vs `{ref_arm}` (variant `all`, >= 200 counts combined)\n")
        L.append("| gene | name | biotype | locus | gpu | " + ref_arm + " | log2(gpu/ref) |\n|---|---|---|---|---:|---:|---:|")
        for j in order:
            i = keep[j]
            nm, bio, loc = info.get(ids[i], ("-", "-", "-"))
            L.append(f"| {ids[i]} | {nm} | {bio} | {loc} | {gpu[i]:.0f} | {ref[i]:.0f} | {ratio[j]:+.2f} |")
    text = "\n".join(L) + "\n"
    with open(out_md, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
