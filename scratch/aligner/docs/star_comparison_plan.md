# Plan: GPU aligner vs STAR, alignment -> featureCounts

Status: **plan only, nothing below has been run.** Facts marked (measured) come from this repo's
data; everything else is a design decision or an assumption to confirm.

## 1. Questions the study should answer

1. **Alignment**: how do speed, memory, mapping rate, pairing and splice-junction calls of the GPU
   aligner compare with STAR, with and without annotation?
2. **Quantification**: does the choice of aligner change gene-level counts from `featureCounts`
   enough to matter? Where do they differ, and which one is closer to the truth?
3. **Cost**: end-to-end time from FASTQ to a count table, and what limits each pipeline.

The headline comparison is only fair if the aligners get the same information. The GPU aligner uses
**no annotation** (it discovers junctions from the reads), so the primary STAR arm is also
annotation-free. STAR with an annotated index is reported as an upper bound, not the baseline.

## 2. Tools (all downloaded into `data/tools/`, never installed system-wide)

| tool | version | source | notes |
|---|---|---|---|
| STAR | 2.7.11b | GitHub release `STAR_2.7.11b.zip` (prebuilt static Linux binary; URL checked, HTTP 200) | |
| featureCounts | Subread 2.0.6 | SourceForge `subread-2.0.6-Linux-x86_64.tar.gz` (URL checked, HTTP 200) | counts SAM or BAM directly |
| minimap2 | 2.28 | already present | optional third arm; cannot do spliced paired-end (single-end style only) |
| samtools | n/a | not installed | avoid needing it: STAR can write `BAM Unsorted`; featureCounts reads SAM |

Machine (measured): 24 cores, 27 GB RAM, 623 GB free on `/root` overlay (use `/root/aligner-data`;
the repo's `C:\` mount is nearly full). STAR on a 100 Mbp genome needs ~2-3 GB RAM, so it fits.

## 3. Datasets

**A. Real**: SRR10065383, 13.2 M pairs x 101 bp, N2 wild type, already on disk.
Library is **unstranded** (measured: mate 1 on the gene strand 49.7%) -> `featureCounts -s 0`.
No truth; used for tool-vs-tool agreement and for timing at realistic scale.

**B. Simulated with gene-level truth** (needs work, see section 6): ~10 M pairs, 2 x 100 bp.
The existing simulator draws only from protein-coding transcripts and records no gene identity.
Extend it to draw from **all** transcripts (piRNA/ncRNA/pseudogene included, because 35% of
the 46,926 genes overlap another gene, 21% on the opposite strand; measured) and to emit the
true gene per fragment plus a truth count table. Add an optional ~5-10% nascent/intronic
fraction so "unassigned" categories behave like real data. Truth is therefore known at three
levels: locus, junctions, gene.

## 4. Arms

| arm | description |
|---|---|
| **GPU** | default 2-stage (junction discovery on the first 2 M pairs, then main pass) |
| **STAR-denovo** | index without GTF, 1-pass |
| **STAR-2pass** | index without GTF, `--twopassMode Basic` (closest analogue to the GPU's 2 stages) |
| **STAR-annotated** | index built with the GTF (`--sjdbGTFfile`, `--sjdbOverhang 100`); upper bound |
| minimap2 (optional) | `-ax splice:sr`, single-end style |

STAR parameters to match the GPU aligner's design where it makes sense: `--runThreadN 24`,
`--genomeSAindexNbases 12` (log2(100 M)/2 - 1), `--alignIntronMax 40000` and
`--alignMatesGapMax 40000` (the GPU aligner's maximum intron is 40 kb; also run STAR defaults and
report both), `--outSAMattributes NH HI AS nM`, `--outSAMtype SAM` for the featureCounts feed.
Everything else default, listed in the report.

## 5. Counting

`featureCounts -p --countReadPairs -s 0 -T 24 -a Caenorhabditis_elegans.WBcel235.113.gtf -t exon -g gene_id`

Variants, run for every arm so the comparison is not an artifact of one counting policy:

1. **default**: unique only (multi-mappers excluded via `NH`), overlaps = ambiguous/unassigned.
2. **`-O --fraction -M`**: multi-overlap and multi-mappers shared fractionally.
3. **protein-coding GTF subset**: removes most overlapping-gene ambiguity.
4. `-R CORE` on the simulated data to get per-read gene assignments, so assignment accuracy can
   be scored against truth, not only the totals.

Cross-check: STAR's own `--quantMode GeneCounts` against featureCounts on STAR's alignments.

## 6. Work needed before anything can run

| item | why |
|---|---|
| **`NH:i` tag in the GPU aligner's SAM** (and `HI`) | featureCounts decides "multi-mapping" from `NH`; without it every GPU read counts as unique. We already have a second candidate locus per read; define NH from candidate score ratio, same rule as MAPQ. Must be byte-checked to leave existing fields unchanged |
| simulator: gene id + truth counts + all biotypes + optional nascent reads | gene-level truth |
| `tools/get_star.sh`, `tools/get_subread.sh` | isolated downloads, pinned versions |
| `tools/run_compare.sh` | one script: index (timed), align (timed, `/proc` peak RSS), count (timed), per arm, 3 repeats |
| `tools/compare_counts.py` | metrics in section 7 |
| confirm featureCounts accepts SEQ/QUAL = `*` and name-grouped unsorted SAM | small pilot on 200 k pairs before the full run |

## 7. Metrics

**Alignment (real and simulated)**
- wall time, CPU-seconds, peak RSS, VRAM; index build time reported separately
- mapped %, MAPQ/NH distribution, proper pairs, soft-clip distribution, spliced %
- junction sets: overlap between arms and with the annotation, novel junction counts, motif mix
- simulated truth: locus, exact junction set, proper-pair correctness (existing `score_sam.py`)
- per-mate concordance between arms, and a sample of discordant reads classified by cause

**Quantification**
- featureCounts summary: assigned / no feature / ambiguous / multimapping / unmapped, per arm
- simulated truth: per-gene error vs the true count (log-scale RMSE, Spearman/Pearson, relative
  error for genes with >= 10 true fragments, bias against expression level and gene length),
  per-read gene-assignment accuracy
- real data: gene-level agreement between arms (log CPM correlation, share of genes within 10%
  and 2x, top discordant genes with the reason: overlapping genes, gene families, short exons,
  intron retention)
- how many genes would change significance is out of scope (no replicates/design); at most note
  the largest fold-changes between arms

## 8. Fairness and pitfalls

- GPU time is noisy because the card is shared (observed 6.8-8.6 s for the same kernels); run each
  arm 3x, report median and range. Keep the page cache warm for all arms and say so.
- Output formats differ: STAR's SAM carries SEQ/QUAL, ours writes `*` (about 4x smaller). Report
  align time with output on and, for STAR, also with `--outSAMtype BAM Unsorted`.
- The GPU aligner's startup (genome load + index, ~3 s) happens every run; STAR loads a prebuilt
  index. Show both "index build" and "per-run" costs.
- 35% overlapping genes + unstranded library means a large ambiguous fraction regardless of
  aligner; compare assigned fractions only within a counting variant.
- STAR is not truth. On real data "differs from STAR" is not "wrong".
- Parameter parity is imperfect (seeding, scoring, multimapper policy). List every flag.

## 9. Proposed sequence

1. Tools + pilot on 200 k simulated pairs: all arms, all counting variants, fix plumbing (NH,
   formats). Expected a few hours of work.
2. Simulator extension and a 10 M-pair truth set (single-threaded Python, ~10 min; parallelise
   if needed).
3. Full simulated comparison.
4. Real-data comparison and timing.
5. Write-up in `README.md` plus `docs/`; publish as a shareable page if you want it.

Rough compute: STAR index ~1-2 min; STAR on 13 M pairs at 24 threads a few minutes; featureCounts
a minute or two per arm; the whole study is hours of wall time, mostly my engineering, not waiting.

## 10. Decisions for you

1. OK to download STAR and Subread binaries into `data/tools/` (same approach as minimap2)?
2. Primary arm: STAR de novo + 2-pass (my recommendation, matches what the GPU aligner knows), or
   annotated STAR as the main comparator?
3. Simulation: include nascent/intronic reads and non-coding transcripts (more realistic, lower
   assigned fraction), or keep it clean (protein-coding only)? I recommend realistic.
4. Counting scope: all genes, protein-coding only, or both (recommended: both)?
5. Should `NH` for the GPU aligner stay a simple two-candidate rule, or do you want true
   multi-mapper reporting (several alignments per read) as part of this study?
