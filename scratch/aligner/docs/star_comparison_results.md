# GPU aligner vs STAR, alignment -> featureCounts: results

Plan: `star_comparison_plan.md`. Raw generated reports and timings are in `results/`.
Everything here was run on one machine (24 cores, RTX 3080 shared with other processes,
27 GB RAM) with STAR 2.7.11b and featureCounts 2.0.6 (built from source; the prebuilt binary
segfaults in this container).

## Setup actually used

| | |
|---|---|
| Reference / annotation | *C. elegans* WBcel235, Ensembl release 113 GTF (46,926 genes; 35% overlap another gene) |
| **Simulated data** | 10 M pairs, 2 x 100 bp, fragments 250 +- 50 bp, all gene biotypes, lognormal expression (sigma 2), several isoforms per gene, 8% nascent (unspliced) fragments, 0.5% substitutions / 0.05% indels / 0.1% N. Truth known per fragment (locus, junctions, gene) |
| **Real data** | SRR10065383, *C. elegans* N2, 13.2 M pairs x 101 bp, unstranded (mate 1 on the gene strand 49.7%) |
| Arms | **GPU** (no annotation, 2-stage) - **STAR annotated** (primary comparator, GTF in the index) - STAR annotated + 40 kb intron/mate-gap limits - STAR **de novo** 2-pass with 40 kb limits (no annotation, the closest analogue of what the GPU aligner knows) |
| Counting | `featureCounts -p --countReadPairs -s 0 -t exon -g gene_id`. Variants: `all` (default: unique, overlaps = ambiguous), `frac` (`-O -M --fraction`), `pc` (protein-coding GTF only), `core` (per-read assignments) |
| Oracle | the simulator's perfect alignments, counted with the same featureCounts settings, so every difference is attributable to the aligner |

Why the oracle matters: only ~94.5% of fragments are assignable even with perfect alignments
(2.3% nascent/intronic, 3.2% in overlapping genes), so "assigned %" must be compared to the oracle,
not to 100%.

## 1. Speed and resources

Alignment only, 10 M simulated pairs, **three repetitions** (wall time includes each tool's own
start-up: the GPU aligner loads the genome and builds its index every run, ~3 s; STAR loads a
prebuilt one):

| aligner | wall, median (range) | CPU-seconds | peak host RAM |
|---|---|---:|---:|
| **GPU aligner** | **16.0 s** (15.8-18.4) | 39 | 1.5 GB (+ ~3 GB VRAM) |
| STAR annotated | 51.3 s (50.3-54.5) | 1,032 | 4.7 GB |
| STAR annotated, 40 kb limits | 37.3 s (37.0-37.8) | 741 | 4.7 GB |
| STAR de novo 2-pass | 90.0 s (90.0-90.4) | 1,534 | 7.8 GB |

STAR needs a one-off index (97 s annotated, 52 s de novo; 1.2 / 1.0 GB on disk).
Real data (13.2 M pairs, one run each): GPU 24.8 s, STAR annotated 68 s, STAR 40 kb 66 s,
STAR de novo 2-pass 121 s. featureCounts (`all` variant) took 12.8 s on the GPU aligner's SAM and
22-26 s on STAR's; STAR's SAM is ~4x larger because it carries SEQ/QUAL, which is the likely
(untested) reason.

Caveats: GPU timings are sensitive to other processes on the shared card. One run in the main
comparison took 40 s (kernels 20.8 s) instead of 16 s while the card was busy; the repeats above
were run for that reason, and the first run is kept in `results/` for transparency. The GPU
aligner writes `*` for SEQ/QUAL; STAR writes them, so its SAM is ~4x larger.

## 2. Alignment accuracy (simulated truth, 10 M pairs)

Primary alignments only. STAR omits unmapped reads from its SAM, so percentages below are over
the reads each tool emitted (STAR emitted 9,979,930 pairs; its own log: 97.85% unique + 1.95%
multi + 0.05% unmapped).

| | GPU | STAR annotated | STAR 40 kb | STAR de novo 2-pass |
|---|---:|---:|---:|---:|
| mates at the correct locus | 98.81% | **99.29%** | 99.30% | 99.27% |
| both mates correct (pairs) | 98.61% | **99.29%** | 99.29% | 99.26% |
| spliced mates, exact junction set | 87.6% | **94.2%** | 94.2% | 93.8% |
| unspliced mates, no spurious junction | 98.53% | **99.17%** | 99.18% | 99.14% |
| wrong locus among MAPQ >= 10 | 726 of 18.9 M | 1,426 of 19.6 M | 1,105 of 19.6 M | 1,683 of 19.5 M |

STAR is ahead on locus and especially on junctions. Most of the junction gap is annotation:
the annotated index lets STAR place short overhangs exactly, and the de novo STAR arm
(94.2% -> 93.8%) is close to annotated STAR, so the GPU aligner's remaining 6-7 point junction gap
is mostly algorithmic (seed-and-extend with a 2-candidate model vs STAR's exhaustive alignment),
not only annotation. Confident (MAPQ >= 10) calls are about equally reliable for all.

## 3. Gene counts vs the oracle (simulated)

Default policy (`all`): all genes, unique reads, ambiguous overlaps unassigned.

| | GPU | STAR annot | STAR 40 kb | STAR de novo 2p |
|---|---:|---:|---:|---:|
| Pearson (log2 count) | 0.9597 | **0.9738** | 0.9732 | 0.9716 |
| Spearman | 0.9616 | **0.9756** | 0.9753 | 0.9736 |
| RMSE (log2) | 0.831 | **0.660** | 0.668 | 0.689 |
| genes (oracle >= 10) within 10% | 92.5% | **95.7%** | 95.7% | 95.2% |
| genes off by > 2x | 4.02% | **2.20%** | 2.20% | 2.32% |
| L1 error (share of fragments misplaced) | 3.13% | **2.07%** | 2.00% | 2.28% |
| fragments counted / oracle | 9.16 M / 9.45 M | 9.26 M | 9.27 M | 9.24 M |

The `frac` variant (multi-mappers shared) is closer for all and keeps the same order
(Pearson 0.984 GPU vs 0.991 STAR; L1 2.26% vs 1.45%). The protein-coding-only variant behaves like
`all`. Full tables for all variants are in `results/rs_10m_report.md`.

**Where the difference comes from (per-fragment, simulated).** Both aligners assign almost
every fragment to the *correct* gene: wrong-gene assignments are 43 (GPU) vs 221 (STAR annotated)
out of ~9.2 M. The entire gap is fragments **discarded as multi-mapping**: the oracle assigns them,
but featureCounts excludes them because the aligner reported `NH > 1`. The GPU aligner discards
2.9% of fragments that way, STAR annotated ~2%, and STAR de novo a bit more than annotated STAR.
(STAR's fragment totals are inflated, 10.25 M for 10 M real fragments, because it writes one record per
multi-mapper locus and featureCounts counts each.) So the practical difference between the two on
this dataset is a ~1-point difference in how aggressively reads are called multi-mapping, not
wrong-gene assignments. (On the real data the discordant genes are mostly gene families and
pseudogenes, section 4; I did not break the simulated loss down by gene class.)

## 4. Real data (no truth; STAR annotated as the reference, not as truth)

STAR's own log: 95.25% uniquely mapped, 1.51% multi, **3.21% unmapped as "too short"** (its
rule: at least 66% of the *pair* length must align). The GPU aligner maps 99.8% of mates.

| | GPU | STAR annot | STAR 40 kb | STAR de novo 2p |
|---|---:|---:|---:|---:|
| spliced reads | 8.08 M (30.7%) | 8.35 M (32.7%) | 8.35 M | 8.37 M |
| distinct introns / annotated | 113.8 k / 72.7% | 100.8 k / 83.4% | 97.4 k / 86.3% | 96.7 k / 85.8% |
| intron observations on annotated introns | 98.96% | 99.44% | 99.53% | 99.18% |
| introns with no splice motif | 6.9% | 4.6% | 3.6% | 3.6% |
| featureCounts `all`: Assigned | 95.54% | 94.43% | 94.59% | 94.13% |
| featureCounts `all`: MultiMapping | 2.02% | 3.88% | 3.72% | 4.17% |
| insert size, median (IQR) | 154 (127-207) | 156 (129-209) | 156 | 156 |

**Gene counts, GPU vs STAR annotated:** Pearson 0.9979, Spearman 0.9982, 96.5% of genes within
10%, 0.45% off by > 2x, L1 difference 3.1%. For comparison, STAR 40 kb and STAR de novo differ
from annotated STAR by only 0.07% and 0.23% (L1): the three STAR configurations are
nearly interchangeable for counting, and the GPU aligner is the one that deviates measurably.

**Why the GPU totals are higher (12.64 M vs 12.34 M fragments, +2.4%).** Per-fragment cross-tab
(`results/SRR10065383_gpu_vs_star_fragments.md`): 92.75% of all fragments are assigned to the same
gene by both and "different gene" is negligible; 344,000 fragments (2.6%) are assigned by the GPU
aligner but are *absent* from STAR's output. Profile of the 819,540 mates that only the GPU aligner
maps: 95.3% have MAPQ >= 10, 77.5% are in proper pairs, only 12% are soft-clipped by >= 20 bases, mean
NM 0.45 vs 0.20 for shared mates. They look like genuine but imperfect reads (e.g. one good mate and
one poor one) that STAR's pair-length rule rejects. Whether including them is better is a policy
question the data cannot settle without truth.

**Most discordant genes** (`results/SRR10065383_report.md`) are almost all pseudogenes and
paralogous families (rpl-37, col-166/167, act-1, npl-4.1/4.2, sams-3), where the GPU aligner
reports fewer unique counts: these are exactly the genes where deciding "which copy" decides the
count. The likely cause is that annotation-aware STAR resolves near-identical copies differently
and calls fewer of those reads multi-mapping; I did not test this directly.

## 5. Findings

1. **Speed**: on 10 M pairs (medians of 3) the GPU aligner is 2.3x faster than the fastest STAR
   configuration (annotated, 40 kb limits), 3.2x faster than annotated STAR at default limits and
   5.6x faster than STAR de novo 2-pass, using 19-39x less CPU time and about a third of the host
   memory (plus ~3 GB of VRAM). STAR also needs a 1-2 minute one-off index. With featureCounts
   included the end-to-end gap narrows, since counting costs 11-26 s for every arm.
2. **Accuracy**: STAR annotated is more accurate on this simulation: +0.5 points of correct locus,
   +6.6 points of exact junctions, and about a third less gene-count error (L1 2.07% vs 3.13%).
   The count error is not wrong-gene assignment but fragments excluded as multi-mapping.
   Annotation is part of STAR's advantage but not all of it: de novo STAR is within 0.4 points of
   annotated STAR on junctions, while the GPU aligner is 6.2 points behind de novo STAR.
3. **Counting impact**: on real data the choice among STAR settings changes gene counts by < 0.3%;
   swapping in the GPU aligner changes them by ~3% (L1) with r = 0.998, concentrated in
   pseudogenes and paralog families. For most differential-expression purposes that is small;
   for those gene families it is not.
4. **Mapping policy differs**: the GPU aligner maps 3.2 points more of the real reads (STAR drops
   pairs failing its 66% rule). That raises counts by 2.4% and is a design choice to make
   explicit, not an unqualified advantage.

## 6. Limitations of this study

- One dataset per type, one organism with a compact genome (100 Mbp, short introns). No claim
  extends to mammalian genomes.
- The simulator is mine: error profile, expression model and fragment lengths are simple, and the
  GPU aligner's parameters were tuned on earlier simulations from the same generator. The real-data
  agreement numbers do not share that bias, but have no truth.
- The GPU aligner does not use annotation and has no multi-alignment output (only 2 candidates per
  read), so its multi-mapper handling is a heuristic (`NH=2` when another locus scores within 1 of
  the chosen alignment, unless pairing resolves it). STAR is configured with defaults apart from the
  intron / mate-gap limits and the 2-pass option.
- featureCounts was only run on unsorted SAM text; no BAM / sorting cost is included for either
  side, and the GPU aligner's SAM omits SEQ/QUAL.
- Timing of the GPU aligner depends on a shared card; medians of three runs are reported for the
  simulated data, a single run for the real data.

## 7. Suggested next steps

1. Tighten the GPU aligner's multi-mapper rule (this alone would close most of the count gap) and
   add a minimum-aligned-fraction option so its mapping policy can match STAR's when wanted.
2. Optional annotation support in the junction stage (it already has a junction table; loading the
   GTF junctions into it is small) to quantify how much of the junction gap annotation explains.
3. A second organism / larger genome to see whether the speed ratio and the accuracy gap hold.
