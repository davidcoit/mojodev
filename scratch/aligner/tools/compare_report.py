#!/usr/bin/env python3
"""Build a markdown report from a run_compare.sh output directory.

Usage: compare_report.py CMPDIR TRUTH.tsv [OUT.md]
  CMPDIR      directory written by run_compare.sh (one subdirectory per arm, incl. `truth`)
  TRUTH.tsv   simulator truth (read mate chrom strand blocks gene kind); for locus/junction scoring

Sections: timings, alignment accuracy vs truth, featureCounts category fractions, gene-count
agreement with the oracle (`truth` arm counted with the same policy), per-fragment gene-assignment
agreement with the oracle.  Run with `python3 -I`.
"""
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["all", "frac", "pc"]
SUMMARY_KEYS = ["Assigned", "Unassigned_NoFeatures", "Unassigned_Ambiguity", "Unassigned_MultiMapping",
                "Unassigned_Unmapped", "Unassigned_Singleton", "Unassigned_Chimera"]


def read_counts(path):
    ids, counts = [], []
    with open(path) as f:
        next(f)
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            ids.append(c[0])
            counts.append(float(c[6]))
    return ids, np.array(counts)


def read_summary(path):
    d = {}
    with open(path) as f:
        next(f)
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            d[k] = float(v)
    return d


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ranks for ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def count_metrics(arm, oracle):
    a, o = np.asarray(arm), np.asarray(oracle)
    keep = (a > 0) | (o > 0)
    la, lo = np.log2(a[keep] + 1), np.log2(o[keep] + 1)
    pear = float(np.corrcoef(la, lo)[0, 1])
    spear = float(np.corrcoef(ranks(a[keep]), ranks(o[keep]))[0, 1])
    rmse = float(np.sqrt(np.mean((la - lo) ** 2)))
    hi = o >= 10
    rel = np.abs(a[hi] - o[hi]) / o[hi]
    return {
        "genes_counted": int(keep.sum()), "pearson_log2": pear, "spearman": spear, "rmse_log2": rmse,
        "genes_ge10": int(hi.sum()), "median_rel_err": float(np.median(rel)),
        "within_10pct": float(np.mean(rel <= 0.10)), "off_by_2x": float(np.mean(
            (a[hi] > 2 * o[hi]) | (a[hi] < 0.5 * o[hi]))),
        "L1_frac": float(np.abs(a - o).sum() / max(o.sum(), 1)), "total": float(a.sum()), "oracle_total": float(o.sum()),
    }


def core_path(armdir):
    return os.path.join(armdir, "aligned.sam.featureCounts")


def load_core(path):
    d = {}
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            d[c[0]] = (c[1], c[3] if len(c) > 3 else "")
    return d


def assignment_agreement(armdir, oracle):
    cats = Counter()
    with open(core_path(armdir)) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            o = oracle.get(c[0])
            if o is None:
                continue
            a_ok = c[1] == "Assigned"
            o_ok = o[0] == "Assigned"
            if o_ok and a_ok:
                cats["same gene" if c[3] == o[1] else "different gene"] += 1
            elif o_ok:
                cats["oracle assigned, arm " + c[1]] += 1
            elif a_ok:
                cats["arm assigned, oracle " + o[0]] += 1
            else:
                cats["both unassigned"] += 1
    return cats


def main():
    cmp_dir, truth_tsv = sys.argv[1:3]
    out_md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(cmp_dir, "report.md")
    arms = [d for d in sorted(os.listdir(cmp_dir)) if os.path.isdir(os.path.join(cmp_dir, d))]
    order = ["truth", "gpu", "star_annot", "star_annot40k", "star_denovo2p"]
    arms = [a for a in order if a in arms] + [a for a in arms if a not in order]
    L = []

    # ---- timings
    L.append("## Timings\n")
    L.append("| step | wall (s) | CPU (s) | peak RSS (GB) |\n|---|---:|---:|---:|")
    tj = os.path.join(cmp_dir, "timings.jsonl")
    if os.path.exists(tj):
        for line in open(tj):
            r = json.loads(line)
            if ".truth." in r["label"]:
                continue
            L.append(f"| {r['label'].split('.', 1)[1]} | {r['wall_s']} | {r['cpu_s']} | {r['maxrss_gb']} |")

    # ---- alignment accuracy vs truth
    L.append("\n## Alignment accuracy vs simulated truth\n")
    for a in arms:
        if a == "truth":
            continue
        sam = os.path.join(cmp_dir, a, "aligned.sam")
        if not os.path.exists(sam):
            continue
        res = subprocess.run(["python3", "-I", os.path.join(HERE, "score_sam.py"), sam, truth_tsv],
                             capture_output=True, text=True)
        L.append(f"**{a}**\n\n```\n{res.stdout.strip()}\n```\n")

    # ---- featureCounts categories
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

    # ---- gene count agreement vs oracle
    L.append("\n## Gene-count agreement with the oracle (truth alignments counted the same way)\n")
    for v in VARIANTS:
        op = os.path.join(cmp_dir, "truth", f"counts_{v}.txt")
        if not os.path.exists(op):
            continue
        ids, oracle = read_counts(op)
        L.append(f"\n**variant `{v}`**\n")
        L.append("| arm | genes | Pearson (log2) | Spearman | RMSE log2 | genes oracle>=10 | median rel err | within 10% | off by >2x | L1 error | total vs oracle |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a in arms:
            if a == "truth":
                continue
            p = os.path.join(cmp_dir, a, f"counts_{v}.txt")
            if not os.path.exists(p):
                continue
            ids2, cnt = read_counts(p)
            assert ids2 == ids
            m = count_metrics(cnt, oracle)
            L.append(f"| {a} | {m['genes_counted']} | {m['pearson_log2']:.4f} | {m['spearman']:.4f} | {m['rmse_log2']:.3f} | "
                     f"{m['genes_ge10']} | {100 * m['median_rel_err']:.2f}% | {100 * m['within_10pct']:.1f}% | "
                     f"{100 * m['off_by_2x']:.2f}% | {100 * m['L1_frac']:.2f}% | {m['total']:.0f} / {m['oracle_total']:.0f} |")

    # ---- per-fragment assignment agreement
    cp = core_path(os.path.join(cmp_dir, "truth"))
    if os.path.exists(cp):
        oracle = load_core(cp)
        L.append("\n## Per-fragment gene assignment vs the oracle (variant `core` = default policy)\n")
        for a in arms:
            if a == "truth" or not os.path.exists(core_path(os.path.join(cmp_dir, a))):
                continue
            cats = assignment_agreement(os.path.join(cmp_dir, a), oracle)
            tot = sum(cats.values())
            L.append(f"\n**{a}** ({tot} fragments)\n")
            L.append("| outcome | fragments | share |\n|---|---:|---:|")
            for k, v in cats.most_common():
                L.append(f"| {k} | {v} | {100 * v / tot:.3f}% |")

    text = "\n".join(L) + "\n"
    with open(out_md, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
