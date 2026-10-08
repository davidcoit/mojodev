# GPU splice-aware nucleotide aligner (Mojo)

A prototype that aligns RNA-seq reads to a genome on the GPU: **4-bit nucleotide
type → minimizer index → seed/chain → splice-aware breakpoint refinement**, all
in Mojo 1.1 on an RTX 3080. Test reference is *C. elegans* WBcel235 (100.3 Mbp);
test reads are simulated spliced reads with known truth.

Original brief: `aligner-instructions.txt`.

## Quick start

```bash
cd scratch/aligner
./setup.sh            # venv with max.gpu, genome + GTF, prebuilt minimap2 (once)
./run.sh 200000       # build, simulate 200k reads, align, score
```

`run.sh` pins `CUDA_VISIBLE_DEVICES=0`. This container exports it as `all`, which
is not valid for CUDA and makes `cuInit` fail with error 100 (no device) even
though `nvidia-smi` works.

Direct use: `./build_main REF.fa READS.fastq OUT.sam [batch=200000] [passes=2] [min_junction_support=1]`

## Layout

| file | role |
|---|---|
| `src/nt4.mojo` | 4-bit encoding (A=0 C=1 G=2 T=3 N=4, codes 5–15 reserved for IUPAC), 2 bases/byte, pointer helpers usable on host and device |
| `src/minimizer.mojo` | canonical (k=15, w=8) minimizer scanner; the *same* function indexes the genome and seeds reads |
| `src/index.mojo` | FASTA → packed genome; minimizer index as a counting-sorted bucket table (2^24 buckets) |
| `src/kernels.mojo` | the GPU kernel: one thread per read |
| `src/junctions.mojo` | pass-1 junction collection, radix sort, for the second pass |
| `src/main.mojo` | driver: parse FASTQ → pack → batches → 2 passes → SAM |
| `src/test_nt4.mojo` | unit tests (packing, canonical-minimizer symmetry, density, N handling) |
| `tools/simulate_reads.py` | spliced read simulator from GTF (errors, indels, Ns, 36% spliced) |
| `tools/score_sam.py` | scores a SAM against simulation truth |
| `tools/bench_minimap2*.sh` | CPU baseline |

## How the aligner works

Per read (one GPU thread each):

1. **Seed**: minimizers of the read, looked up in the genome index (bucket →
   hash match). Minimizers with >100 genome hits are skipped.
2. **Chain**: anchors are insertion-sorted by (strand, ref) and chained by DP.
   The gap cost separates indels (|Δdiag| ≤ 12, cost grows with size) from
   introns (20 ≤ Δdiag ≤ 40 kb, flat cost), which is what makes it splice-aware.
3. **Refine each breakpoint**: between two anchors on different diagonals, scan
   the surrounding query window for the split that maximizes matches, with a
   +2 bonus for canonical `GT..AG` / `CT..AC` introns. Emits `I`, `D` or `N`.
4. **Ends**: ungapped X-drop extension, then soft clip. A clipped or
   over-extended end is re-placed across an intron: first via known junctions
   (pass 2, overhang ≥ 3 bp), else by a de novo search for a canonical intron
   within 10 kb (overhang ≥ 8 bp).
5. **MAPQ** from the best chain vs the best chain *outside its own reference
   span*, so tandem duplicates count as competition.

**Two-pass (STAR-style)**: pass 1 aligns de novo; junctions seen in unique
alignments with ≥12 bp flanks are tabulated and pass 2 uses them to place short
overhangs that de novo search cannot do safely.

**Memory**: genome 50 MB (4-bit) + index 246 MB. For 5M reads about 3.1 GB of
VRAM is allocated (reads 640 MB, output records 960 MB, per-batch scratch 1.2 GB).
Everything fits in the ~7 GB free on the 10 GB card (3 GB is held by other
processes). No pseudoalignment was needed; the whole genome is indexed.

## Results (C. elegans, 100 bp simulated reads, 36% spliced, 0.5% subs, 0.05% indels, 0.1% N)

5,000,000 reads, final build:

| stage | time |
|---|---|
| reference load + 4-bit pack | 0.8 s |
| minimizer index (host, 22.3 M minimizers) | 2.2 s |
| FASTQ read + parse + pack (1.07 GB) | ~6.6 s (3.7–4.7 s is file read) |
| **GPU pass 1 (de novo)** | **2.2–2.9 s** (1.7–2.3 M reads/s) |
| **GPU pass 2 (junction DB)** | **0.9–1.2 s** (4.2–5.6 M reads/s) |
| SAM write | 2.0 s |

GPU timings vary between runs because the card is shared with other processes.

| | minimap2 2.28 `splice:sr`, 24 threads | this aligner |
|---|---|---|
| alignment time, 5M reads | 161 s (+4 s index) | 3.1–4.1 s GPU, ~16 s end-to-end |
| mapped | 99.22% | 99.99% |
| spliced reads, exact junction set | 66.1% | **87.7%** |
| spliced reads, locus correct | 97.1% | 99.1% |
| unspliced reads, locus correct | 98.6% | 98.4% |
| wrong locus among MAPQ ≥ 10 | 121 / 4.85 M | 69 / 4.80 M |

On 200k reads the same ordering holds (86.0% vs 66.1% exact junctions; 0.15 s GPU for both passes vs 10.4 s for minimap2 at 24 threads, which includes its index build).

### Read these numbers with care

- **The accuracy comparison favours this aligner.** Reads were simulated from
  annotated protein-coding transcripts whose introns are almost all canonical;
  the motif bonus exploits that, and parameters were tuned on this simulation.
  Real data has non-canonical introns, intron retention, unannotated isoforms,
  and quality-dependent errors. Expect the gap to be smaller on real reads.
- **minimap2 is not the fastest CPU spliced aligner.** STAR and HISAT2 are faster
  on short reads; they were not run (not installed), so the speedup over the best
  CPU tool is smaller than shown.
- "Exact junctions" is strict. Reads with ≤4 bp overhangs are essentially
  unplaceable without annotation: ~9% of spliced reads still fail this way.
- Remaining wrong-locus reads are almost all MAPQ < 10 (tandem paralogs).
- Truth for reads containing an indel error is shifted by 1 bp, so the `blocks`
  column understates accuracy; `junctions` is the meaningful column.

## Mojo 1.1 notes (what changed vs older docs)

- GPU modules are in the **`max`** package: `from max.gpu.host import DeviceContext`,
  `from max.gpu import global_idx`. `pip install max==26.6.0` into a venv; the system
  Mojo install has no `gpu` module. `fn` is gone (use `def`); `out` is a reserved
  parameter name.
- Kernel scalar arguments must be fixed-width (`Int32`, not `Int`).
- `enqueue_function[kernel](bufs..., grid_dim=, block_dim=)`: the grid is rounded
  up to whole blocks, so **kernels must bound-check their own thread index
  against the batch size**. Omitting that corrupted scratch memory once here.
- **Pinned host buffers (`enqueue_create_host_buffer`) are extremely slow under
  WSL2** (≈2.8 s for 10 MB). Plain `List` memory copies to the device at full
  speed (100 MB in 26 ms). Index build went 12 s → 2.2 s after switching.
- Host pointers need `.unsafe_origin_cast[MutAnyOrigin]()` to pass to code taking
  `UnsafePointer[T, MutAnyOrigin]`; `List.unsafe_ptr()` also needs `.unsafe_mut_cast[True]()`.
- No `sort` in the stdlib path I could find, and no `InlineArray`; `junctions.mojo`
  has a small radix sort and the minimizer window lives in 8 registers.

## Limitations and next steps

- Reads ≤ 256 bases (fixed 128-byte slots); longer reads are left unmapped.
  Single-end only. Output is SAM with `*` for SEQ/QUAL.
- One thread per read with global-memory scratch; no attempt yet to use shared
  memory, warp-cooperative chaining, or overlap copies with compute.
- The de novo end search (up to 10 kb) is why pass 1 is ~2.4× slower than pass 2.
- Index construction is on the host (2.2 s); it is embarrassingly parallel and
  could move to the GPU. FASTQ parsing is now the largest host cost.
- Not done: paired-end, base qualities, gapped (banded) extension within exons
  beyond the single breakpoint scan, non-canonical splice sites on the de novo
  path, real RNA-seq data, comparison with STAR/HISAT2.

## Log

- **2026-10-08**: built from scratch in worktree `gpu-aligner` (branch
  `worktree-gpu-aligner`). Not under `tutor/work/`, so CLAUDE.md's no-edit rule
  for user Mojo code did not apply. `workingmemory.md` was not updated: it exists
  only as an uncommitted file in the main checkout, not in this worktree. Add an
  entry there pointing to this directory.
