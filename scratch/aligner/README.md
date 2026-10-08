# GPU splice-aware nucleotide aligner (Mojo)

A prototype that aligns single-end or **paired-end** RNA-seq reads to a genome on the
GPU: **4-bit nucleotide type → minimizer index → seed/chain → splice-aware breakpoint
refinement → pair resolution**, all in Mojo 1.1 on an RTX 3080. Test reference is
*C. elegans* WBcel235 (100.3 Mbp). Test reads: simulated spliced reads/pairs with known
truth, plus a real 13.2 M-pair *C. elegans* RNA-seq run (SRR10065383).

Original brief: `aligner-instructions.txt`.

## Quick start

```bash
cd scratch/aligner
./setup.sh            # venv with max.gpu, genome + GTF, prebuilt minimap2 (once)
./run.sh 200000 se    # build, simulate 200k single-end reads, align, score
./run.sh 200000 pe    # same with 200k simulated read pairs
tools/get_real_pe.sh  # (optional) real paired-end run SRR10065383 -> /root/aligner-data
```

`run.sh` pins `CUDA_VISIBLE_DEVICES=0`. This container exports it as `all`, which
is not valid for CUDA and makes `cuInit` fail with error 100 (no device) even
though `nvidia-smi` works.

Direct use:

```bash
./build_main REF.fa OUT.sam READS_1.fastq [READS_2.fastq] \
    [--chunk 500000] [--jn-sample 2000000] [--min-support 1] [--max-frag 60000]
```

FASTQ is streamed in chunks, so input size is not limited by VRAM (plain FASTQ only,
no gzip; reads ≤ 256 bases). With two files the run is paired-end.

## Layout

| file | role |
|---|---|
| `src/nt4.mojo` | 4-bit encoding (A=0 C=1 G=2 T=3 N=4, codes 5–15 reserved for IUPAC), 2 bases/byte, pointer helpers usable on host and device |
| `src/minimizer.mojo` | canonical (k=15, w=8) minimizer scanner; the *same* function indexes the genome and seeds reads |
| `src/index.mojo` | FASTA → packed genome; minimizer index as a counting-sorted bucket table (2^24 buckets) |
| `src/kernels.mojo` | the GPU kernel: one thread per read, up to 2 candidate alignments per read |
| `src/junctions.mojo` | junction counting across chunks, radix-sorted table for the junction-aware pass |
| `src/fastq.mojo` | streaming FASTQ reader that packs bases straight into 4-bit slots |
| `src/samout.mojo` | pair resolution (host) and SAM writer |
| `src/main.mojo` | driver: GPU state, phase A (junction discovery), phase B (stream, align, pair, write) |
| `src/test_nt4.mojo` | unit tests (packing, canonical-minimizer symmetry, density, N handling) |
| `tools/simulate_reads.py`, `tools/simulate_pairs.py` | spliced single-end / paired-end read simulators from a GTF (errors, indels, Ns) |
| `tools/score_sam.py` | scores a SAM against simulation truth (per mate and per pair) |
| `tools/eval_real.py`, `tools/only_mine.py` | truth-free evaluation on real data; mate-by-mate comparison of two SAMs |
| `tools/get_real_pe.sh` | fetch + convert the real run |
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

**Junction-aware second stage (STAR-style)**: phase A aligns the first 2 M templates de
novo; junctions seen in unique alignments with ≥12 bp flanks (kept if canonical, or seen
by ≥3 reads) are tabulated, and the main pass uses them to place short overhangs that
de novo search cannot do safely.

### Paired-end

Mates sit in adjacent GPU slots and are aligned independently, but each read returns
its best chain **and** the best chain at a different locus (2 candidate alignments).
On the host, every (mate 1 candidate, mate 2 candidate) combination is scored:
chain scores + 20 if compatible (same chromosome, opposite strands, forward mate not
past the end of the reverse mate, genomic span ≤ 60 kb). The best combination wins;
the runner-up combination sets a pair-level MAPQ, so a mate that is ambiguous on its
own is resolved by its partner (a mate gets `max(own MAPQ, pair MAPQ)` in a proper
pair). SAM output has correct FLAG bits, RNEXT/PNEXT, and TLEN (genomic span,
introns included). An unmapped mate is placed at its partner's position.

**Memory**: genome 50 MB (4-bit) + index 246 MB, plus about 1.4 GB of per-chunk buffers
at the default 500 k templates (1 M reads: reads 128 MB, 2 candidate records 384 MB,
scratch 1.2 GB). Usage is independent of input size, and everything fits in the ~7 GB
free on the 10 GB card (3 GB is held by other processes). No pseudoalignment was needed;
the whole genome is indexed.

## Results

### Single-end, simulated (100 bp, 36% spliced, 0.5% subs, 0.05% indels, 0.1% N)

5,000,000 reads (measured with the earlier in-memory driver; junction table built from all reads):

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

### Paired-end, simulated (200 k pairs, 2 x 100 bp, fragments 250 +- 50 bp)

| metric | value |
|---|---|
| both mates mapped | 99.99% |
| flagged proper pair | 99.78% |
| both mates at the correct locus | 98.63% (remaining misses are almost all identical tandem duplicates, MAPQ < 10) |
| spliced mates, exact junction set | 86.8% |
| wrong locus among MAPQ >= 10 | 12 of 383,922 mates |

### Paired-end, real: *C. elegans* N2, SRR10065383 (13,185,419 pairs x 101 bp, HiSeq 4000)

Wall-clock for the whole run, FASTQ in, SAM out (RTX 3080, 24-core host):

| stage | time |
|---|---|
| reference + index | ~3 s |
| phase A, junction discovery on first 2 M pairs | 4.4-5.7 s |
| FASTQ parse + 4-bit pack (9.5 GB of text) | 15-21 s |
| **GPU alignment, 26.4 M reads** | **6.8-6.9 s (3.8-3.9 M reads/s)** |
| pair resolution + SAM write (2.1 GB) | 6.7-7.2 s |
| **total** | **38-46 s** |

| | GPU aligner | minimap2 2.28 `splice:sr`, 24 threads |
|---|---|---|
| wall time | 38-53 s | 125 s (1,790 CPU-s) |
| reads mapped | 99.82% | 98.39% |
| MAPQ >= 10 | 96.4% | 96.8% |
| proper pairs | 98.95% | n/a (see below) |
| spliced reads | 8.08 M (30.7%) | 6.27 M (24.2%) |
| intron observations on annotated introns | 98.96% | 99.35% |
| distinct introns / annotated | 113.8 k / 72.7% | 95.2 k / 86.1% |
| no splice motif, all distinct introns | 6.9% | 3.0% |
| insert size (unspliced proper pairs) | median 154, IQR 127-207 | n/a |

- **minimap2 cannot align spliced paired-end reads**: it errors with `--splice and --frag
  should not be specified at the same time`, and `splice:sr` silently treats two files as
  independent single-end reads. The baseline above is that mode, so it has no pairing
  information and its SAM carries SEQ/QUAL (4x larger than mine, which writes `*`).
- Per-mate agreement: of 25.9 M mates mapped by both, 95.5% start at the same position
  (+-5 bp); the rest are mostly soft-clip vs short-overhang differences. 376 k mates are
  mapped only by the GPU aligner (315 only by minimap2); 85% of those carry the proper-pair
  flag and 87% have MAPQ >= 10, consistent with real rescues of clipped or low-quality mates
  by pairing and the junction table, though there is no truth to confirm it.
- **There is no ground truth on real data.** The GPU aligner is more sensitive (more spliced
  reads, more mates mapped) but noisier: 6.9% of its distinct introns lack a splice motif
  against 3.0% for minimap2. The excess is in singleton introns (20.6% motif-less at support 1,
  vs 14.1% for minimap2); introns seen by 5+ reads are 99.3% motif-bearing (minimap2 99.8%).
  The precision/recall trade-off is not settled.
- **Junction precision rule**: a junction at a breakpoint with no GT..AG / GC..AG / AT..AC
  motif (either strand) is only accepted if both flanks are >= 35 bases; otherwise the short
  side is dropped (right tail truncated, then re-placed by the end rescue if a canonical
  intron exists; or a short first exon clipped). On the real run this cut motif-less distinct
  introns from 13.6% to 6.9% and removed ~9 k introns while losing 1 of 82.7 k annotated ones;
  simulated accuracy was unchanged (86.7% exact junctions, 99.78% proper pairs).

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

- Reads <= 256 bases (fixed 128-byte slots); longer reads are left unmapped. SAM writes `*`
  for SEQ/QUAL. Plain FASTQ only (no gzip); `tools/get_real_pe.sh` produces plain FASTQ.
- No mate rescue: if one mate cannot be seeded (0.30% of real pairs) it stays unmapped
  instead of being searched for near its partner. Library strandedness is not used.
- Residual junction noise: singleton introns are still ~21% motif-less (see above). Real
  trans-splicing (SL1/SL2 leaders) and operons also produce legitimate odd junctions in
  *C. elegans*, so some of this may be biology rather than error.
- Host work now dominates a real run: single-threaded FASTQ parse (15-21 s of ~40 s) and
  pairing + SAM (7 s). Overlapping parsing with GPU work, or a multi-threaded parser, would
  cut the wall time roughly in half.
- One thread per read with global-memory scratch; no shared memory or warp-cooperative
  chaining. The de novo end search (up to 10 kb) makes de novo alignment ~2.4x slower than
  the junction-aware pass.
- Index construction is on the host (2 s) and could move to the GPU.
- Not done: base qualities, long reads, comparison with STAR / HISAT2 (not installed),
  differential-expression-grade evaluation of the real run.

## Log

- **2026-10-08**: built from scratch in worktree `gpu-aligner` (branch
  `worktree-gpu-aligner`). Not under `tutor/work/`, so CLAUDE.md's no-edit rule
  for user Mojo code did not apply. `workingmemory.md` was not updated: it exists
  only as an uncommitted file in the main checkout, not in this worktree. Add an
  entry there pointing to this directory.
- **2026-10-08 (later)**: added paired-end support (two-candidate kernel output, host pair
  resolution, streaming FASTQ, junction discovery on a sample), simulated pairs, and the real
  run SRR10065383. Fixed along the way: this checkout is on a nearly full `C:\` mount, so
  large data lives in `/root/aligner-data` (outside the repo).
