# bitte — bitwise transfer entropy for binary sequences

Pure Mojo (1.1.0), no `max`, no GPU, no Python. An experiment in estimating
transfer entropy between two binary time series using bit-packed storage and
almost nothing but AND / XOR / NOT / shift / popcount.

```
bitte.mojo       the single-pair estimator
network.mojo     all-pairs over a network of P signals (shared target masks)
gen.mojo         xorshift64* and the synthetic channels used for testing
validate.mojo    correctness: vs a scalar oracle, and vs closed-form answers
netvalidate.mojo correctness: all-pairs vs single-pair, planted-network recovery
bench.mojo       cost measurements
smoke.mojo       smallest end-to-end check

gpunet.mojo      GPU kernel, all-pairs, k = l = 1 (needs the max package)
gpurun.mojo      GPU driver: verifies against CPU, times, scales in P
```

CPU, with the stock toolchain:

```
mojo build -O3 validate.mojo    -o validate    && ./validate
mojo build -O3 netvalidate.mojo -o netvalidate && ./netvalidate
mojo build -O3 bench.mojo       -o bench       && ./bench
```

GPU. The Mojo stdlib ships only GPU *detection* (`has_accelerator`,
`is_nvidia_gpu`); `DeviceContext`, `barrier`, shared memory and kernel launch
all live in the `max` package, so it needs its own toolchain:

```
python3 -m venv ../.venv-max && ../.venv-max/bin/pip install max==26.6.0
../.venv-max/bin/mojo build -O3 gpurun.mojo -o gpurun && ./gpurun
```

## What it computes

```
TE(X -> Y) = sum p(y', h, g) log2[ p(y'|h,g) / p(y'|h) ]

y' = y[n+1]                        the future target bit
h  = y[n] .. y[n-k+1]              k target history bits
g  = x[n-u+1] .. x[n-u-l+2]        l source history bits at lag u
```

`k`, `l`, `u` are per-variable arguments. Sequences may be any length as long
as source and target match; the usable window is
`n0 = max(k-1, u+l-2) .. n-2`, and everything is driven off one joint count
table `n4[y', h, g]` of `2^(1+k+l)` bins — the three lower marginals are
summations over it, so only one pass is ever needed.

## The idea

Standard estimators walk the time axis and build an integer state index per
sample. Here the time axis never appears in an inner loop.

Pack each sequence one bit per observation. Then **"the j-th history bit, for
every n at once" is just the packed array read at a shifted bit offset** — a
*bit-plane*, extracted with a word-wise funnel shift. The set of time steps
matching a given state is

```
match = P_0^(b0) & P_1^(b1) & ... & P_d^(bd)      P^(1) = P,  P^(0) = ~P
```

and its count is `popcount(match)`: 64 time steps tested per instruction.

Three things make that cheap rather than merely cute:

1. **Share prefixes.** Enumerate the `2^(k+l)` states as a DFS over planes, so
   each node does *one* AND pass over its parent's mask instead of `k+l`
   passes. Per-state cost collapses from `k+l` passes to ~2.

2. **Complement for free.** `mask & ~P == mask ^ (mask & P)`, so once the
   1-child's mask is computed the 0-child is one XOR pass — and needs *no*
   popcount at all, since its count is `parent_count - count_1`. The 1-child's
   mask is still sitting in the level-`d+1` buffer when the 0-child needs it,
   because the subtree below only ever wrote levels `≥ d+2`.

3. **Prune empties.** A node with an empty mask has an all-zero subtree.

Leaves need no extra plane pass either: one AND+popcount against the future
plane splits the matching steps into `y'=1` and `y'=0`.

Cost: `~3 · 2^(k+l) · m/64` word ops, versus `~m · (k+l)` for the scalar walk.
Bitwise should win while `2^(k+l) < ~20(k+l)`, i.e. up to `k+l ≈ 8`.

### Time-blocking, and why it helps more than expected

`count()` walks the whole time axis once per tree node, so at `k+l = 14` it
streams ~2 GB of masks through RAM. `count_tiled(tw)` inverts the loops: take
one tile of `tw` words of the time axis, run the *entire* tree over it, move
on. Mask levels become `8·tw` bytes and stay in L1.

The measured optimum tile is **1 word**, far smaller than cache-sizing would
suggest — because the real win is not cache, it is pruning. One word is 64
consecutive samples, which can contain at most 64 of the `2^(k+l)` history
states, so nearly every subtree is empty. At `k+l = 14` that takes live nodes
to **3.1%** of the tree and the run time from ~1950 ms to ~100 ms, about
**20x**.

`count_auto()` picks: streaming for `k+l ≤ 7`, 2-word tiles at 8, 1-word tiles
above.

### Rotation surrogates, nearly free

A cyclic rotation of the source destroys its timing against the target while
preserving its marginal and its entire autocorrelation — a well-behaved null.
In bit-packed form a rotation is **just a different read offset**: concatenate
the source with itself once (`BitSeq.doubled()`) and every surrogate costs one
re-slice of `l` planes plus a count pass. No shuffling of samples, no
repacking. `rotation_null()` returns mean / sd / p-value, which gives the bias
floor for *this* data rather than for an idealised independent pair.

## Correctness

`./validate` — all checks pass.

- **216 configurations** (`k,l,u ∈ 1..3`, lengths 63/64/65/127/128/129/1000/4097
  to straddle the word boundary) agree with a scalar oracle to **0.0**, i.e.
  bit-identical in double precision. The scalar walk over *packed* bits agrees
  too, and `count_tiled` reproduces the streaming counts exactly at tile widths
  1, 2, 3, 7, 64 and 100000. Counts are checked to sum to `m` every time.
- Closed-form channels at n = 2,000,000:

  | channel | expected | got |
  |---|---|---|
  | perfect copy, lag 1 | 1.0 | 1.000000 |
  | BSC ε=0.10 → 1−H(ε) | 0.531004 | 0.531996 |
  | BSC ε=0.25 → 1−H(ε) | 0.188722 | 0.188834 |
  | severed, ε=0.50 | 0 | 0.000000 |
  | AR(1) target, unrelated source | 0 | 0.000001 |

  The AR case matters: the target is highly predictable *from its own past* and
  TE stays at zero, which is the thing a correlation measure gets wrong.
- **Lag recovery.** Coupling planted at u=3, scanning u=1..6 peaks exactly at
  3 with TE 0.714126 against 1−H(0.05) = 0.713603.
- **Source history length.** For `y[t] = x[t-1] XOR x[t-2]`, either source bit
  alone is uninformative and the pair is decisive: `l=1` → 0.000000,
  `l=2` → 1.000000, `l=3` → no further gain. A real `l` is distinguished from a
  spurious one.
- **Surrogates** separate coupled (p = 0.04) from uncoupled (p = 0.40).

Miller–Madow correction is available via `te_bits_mm()`; on the severed channel
it moves the plug-in 0.000000 to −0.000001.

## Cost (1M observations, best of 3, this WSL2 box)

Packed: 122 KB per sequence. `List[Int]`: 7 MB. That ratio is the point.

| k+l | AND-tree (auto) | scalar/packed | scalar/List[Int] | vs scalar/packed |
|---|---|---|---|---|
| 1+1 | 0.32 ms | 4.63 | 4.99 | **14.6x** |
| 1+2 | 0.66 | 5.54 | 7.10 | **8.4x** |
| 2+2 | 1.37 | 6.68 | 7.32 | **4.9x** |
| 2+3 | 2.90 | 7.67 | 8.46 | **2.6x** |
| 3+3 | 5.71 | 8.48 | 9.65 | **1.5x** |
| 4+3 | 11.4 | 9.73 | 10.6 | 0.85x |
| 4+4 | 22.6 | 10.6 | 11.5 | 0.47x |
| 7+7 | 96.8 | 17.3 | 16.9 | 0.18x |

`scalar/packed` is the honest baseline — it holds the tight representation
fixed and changes only the algorithm.

- **Crossover is `k+l ≈ 7`.** Below it the AND-tree wins, by 15x at `k=l=1`.
  Above it the scalar walk wins, because `2^(k+l)` outgrows the 64x packing
  factor. This is the main result and it is not negotiable by tuning: it is the
  two cost models crossing.
- Throughput is flat in n: ~1.4 ns/observation at `k=l=2` from 10⁴ to
  3.2×10⁷ samples, a steady 4–5x over scalar.
- **Lag scan** u=1..32 at `k=l=2`: 47 ms with `set_lag` reuse (target and
  future planes built once), 56 ms rebuilding per lag, 237 ms scalar — 4–5x
  across runs.
- **Network screen**, 64 candidate sources vs 1 target at `k=l=u=1`: 33 ms vs
  ~325 ms, about **10x**, and it correctly picks the planted source (7) with
  TE 0.277375 against an expected 0.278072. All 64 packed sources are 7.6 MB
  and stay in cache; as `List[Int]` they are 488 MB.
- **Sparse data** (p = 0.02) prunes hard — 0.2% live nodes at `k+l = 16` — which
  keeps the AND-tree at parity out to `k+l ≈ 14` instead of 7.

## Networks of signals (`network.mojo`)

For P signals there are `P(P-1)` ordered pairs, and running the single-pair
estimator `P²` times repeats work. Of the `k+l+1` planes, `k+1` of them — the
future plane and the k target-history planes — depend only on the *target*, and
the plane order already walks those first. So for one target, descend them once
and materialise the `2^(k+1)` masks they produce; every source reuses them.

Those masks are a **partition** of the usable time axis (disjoint, covering),
which buys a second saving: under a given target mask the deepest source level
needs only `c1 = popcount(M & G)` and `c0 = |M| - c1`, so the final complement
mask is never materialised at all — only its count.

Plane passes over the time axis, per ordered pair:

| | per-pair `BitTE` | `NetTE` | |
|---|---|---|---|
| k=l=1 | 12 | **4** | 3.0x |
| k=2, l=1 | 24 | **8** | 3.0x |
| k=1, l=2 | 24 | **16** | 1.5x |
| k=l=2 | 48 | **32** | 1.5x |

Measured at P=64, n=1e6, k=l=1: **2588 ms → 702 ms, 3.7x** (slightly better
than the 3.0x the pass count predicts, since the shared target masks also cut
plane re-slicing). The win is largest when `l` is small, which is the usual
binary-network case.

`./netvalidate` — all checks pass: **10,800 pair estimates** across
`k,l,u ∈ 1..3`, P=5, four tile widths and lengths straddling the word boundary
agree with the single-pair estimator to **4.4e-16**; every count table sums to
`m`; and a planted 8-node network is recovered cleanly (planted edges 0.712 and
0.118 bits, strongest absent edge 1.4e-5). Note the indirect path 0→1→2 shows
**zero** TE at lag 1 and appears only at lag 2, which is correct — lag
specificity, not a missed edge.

## GPU (`gpunet.mojo`, `gpurun.mojo`)

The network case is what a GPU is for: `P²` independent pairs, no cross-pair
communication, and a dataset small enough to sit in L2 because it is bit-packed.

**Mapping: one block per ordered pair.** Each thread strides over the time axis
a word at a time, keeps the whole 3-plane tree in registers, and accumulates
the eight counts of (y', y, x) in eight more registers; the block then folds
its threads through eight shared bins and writes eight `Int32`. Consequences:

- **No global atomics anywhere**, and no barriers in the hot loop. Every
  thread's words are independent because the complement identity *and* the
  count subtraction both hold per word — a thread never needs a neighbour's
  totals.
- It deliberately **gives up** the shared-target-mask trick above (8 passes per
  pair instead of 4). Sharing would need either `P·8` shared bins and `P·8`
  atomics per thread, or global atomics. On a GPU the arithmetic is free and
  the atomics are not, so recomputing per pair is the better trade. This is the
  one place where the best CPU algorithm is *not* the best GPU algorithm.

Measured, P=64, n=1e6, k=l=1, RTX 3080 (verified: GPU counts match the CPU
estimator with worst difference **0.0**, and all 4032 count tables sum to `m`):

| | time | vs GPU |
|---|---|---|
| GPU kernel (best of 5) | **4.05 ms** | — |
| CPU `NetTE` | 702 ms | 173x |
| CPU per-pair `BitTE` | 2588 ms | **640x** |

Upload of all 64 packed signals (7.8 MB) is **4.9 ms**, so the whole analysis is
~9 ms end to end. Scaling in network size, at ~1.0 µs/pair:

| P | pairs | kernel | µs/pair | CPU extrapolated |
|---|---|---|---|---|
| 8 | 56 | 0.12 ms | 2.13 | 0.04 s |
| 32 | 992 | 0.92 ms | 0.93 | 0.6 s |
| 64 | 4032 | 4.09 ms | 1.02 | 2.6 s |
| 128 | 16256 | 16.3 ms | 1.00 | 10.4 s |
| 256 | 65280 | **61.2 ms** | 0.94 | 42 s |

Flat per-pair cost from P=32 (≈1000 blocks) upward — the GPU saturates there
and is then linear in pair count. A 256-node network at 1M samples is 61 ms.

### Two environment traps, both costly

- **Pinned host buffers are catastrophic under WSL2 here.**
  `enqueue_create_host_buffer` + copy took **2257 ms** for 7.8 MB (~0.3 s/MB),
  500x the kernel. Copying straight from `List` memory with
  `ctx.enqueue_copy(buf, lst.unsafe_ptr())` takes **4.9 ms** — a 458x
  difference, and it would have silently dominated every measurement.
- **Kernel scalar args must be fixed-width.** `Int`/`UInt` do not conform to
  `DevicePassable`; the error surfaces as a wall of template-instantiation
  noise ending in "use a fixed-width type such as Int32". Use `Int32`.

### What a GPU does not fix

The `k+l ≈ 7` crossover is arithmetic, not hardware: a GPU accelerates the
AND-tree and the scalar walk by similar factors. GPU raises throughput; it does
not extend the method to larger state spaces.

## Real data: calcium fluorescence (`prep_spikes.py`, `tespikes.mojo`)

`data/fluorescence_iNet1_Size100_CC01inh.txt` — a simulated network of 100 neurons
with inhibition, 179,498 frames at 50 Hz (~1 hour), 122 MB of text. Not tracked
in git; regenerate the packed form with `prep_spikes.py`.

```
../.venv-max/bin/python -I prep_spikes.py \
    data/fluorescence_iNet1_Size100_CC01inh.txt data/spikes_z25.bits --z 2.5
../.venv-max/bin/mojo build -O3 tespikes.mojo -o tespikes && ./tespikes data/spikes_z25.bits
```

**Detector.** A calcium indicator integrates spikes and decays slowly, so the raw
trace is a smeared, drifting spike train and thresholding it detects "was recently
active". The first difference removes the decay and the drift and leaves a sharp
transient per spike. The threshold is per neuron from that neuron's own median and
MAD (`z > 2.5`), so a noisier trace does not look busier, and only the **rising
edge** of each run is kept so one transient gives one event. Result: 178,139 events,
density 0.99% (0.50 Hz), per-neuron rates 0.88–1.08%. Packing round-trip is verified
in-script. 119.8 MB of text becomes 2.19 MB of bits, 56x smaller.

The packed file is byte-for-byte the layout of `BitSeq.w`, so `tespikes.mojo` reads
it straight into buffers with no conversion (header: magic, p, n, wps; then
`p*wps` little-endian `uint64`, signal-major, LSB-first, including BitSeq's guard
word).

**Three lags, all 9,900 ordered pairs.** The null comes free: the signal list handed
to the estimator is `[100 originals..., 100 cyclic rotations...]`, so one all-pairs
run over 200 signals yields both edge `i→j` and its own surrogate `(rotated i)→j`.
A rotation kills the source's timing against the target while preserving its rate
and autocorrelation. The largest of the 9,900 surrogate values is the floor a real
pair must clear.

| delay | frames | strongest pair | null ceiling | pairs above |
|---|---|---|---|---|
| short | 1 (20 ms) | **0.002308** | 0.000065 | 8,076 |
| medium | 5 (100 ms) | 0.000070 | 0.000062 | 1 |
| long | 25 (500 ms) | 0.000068 | 0.000075 | 0 |

All the directed information sits at one frame and is gone by 100 ms, which is the
right shape: at 20 ms per frame a monosynaptic delay is far below one frame, so a
real influence shows at the first lag and nowhere else.

**Timing** (39,800 pairs per lag, since the surrogates ride along):

| stage | time |
|---|---|
| read + threshold 122 MB of text | 2.7 s |
| pack to bits | 0.02 s |
| upload 4.28 MB to GPU | 6.0 ms |
| **all pairs, one lag, GPU** | **27.9 ms** (0.70 µs/pair) |
| all pairs, one lag, CPU `NetTE` | 1,181 ms (29.7 µs/pair) |

GPU is **42x** the optimised CPU path here — well short of the 640x the same kernel
reaches on longer sequences. The reason is in the arithmetic: at 179,497 frames each
thread handles only ~11 words, so launch and scheduling dominate rather than the
ANDs. Fewer, fatter blocks would recover most of it.

**What the numbers do not say.** Worth stating plainly, because the headline count is
misleading on its own:

- **8,076 of 9,900 pairs clear the null at one frame — that is not 8,076 synapses.**
  This culture fires in network-wide bursts (0.3% of frames hold ≥10 simultaneous
  events, one holds 98), and a burst makes every pair mutually predictive whether or
  not they are wired together. Pairwise TE cannot separate a direct connection from
  shared drive. The standard remedy is to condition on low-population-activity
  frames, which this representation makes almost free: it is one more mask ANDed
  into the root of the tree, and `count()` already starts from a root mask.
- **It is not a firing-rate artifact.** Checked: cell 45, the most frequent source
  among the top edges, ranks 83rd of 100 by rate. Per-neuron thresholding leaves all
  rates within 0.88–1.08%, so rate cannot drive the asymmetry.
- **The rising-edge rule empties two of the eight bins.** An event frame cannot be
  followed by another event, so `y'=1, y=1` never occurs; counts for the top pair are
  `[174574, 1360, 1627, 154, 1604, 177, 0, 0]`. Those states contribute nothing in
  either direction, so the estimate rests on the frames where the target was quiet.
  A detector choice, not biology.
- The mechanism behind the top edge, in one line: when cell 45 fires and cell 77 was
  not already firing, cell 77 fires next 177/1537 = **11.5%** of the time, against a
  baseline of **1.00%**. An 11-fold change in the odds, carried by 177 events out of
  179,496 frames — which is why it is only 0.0023 bits.

An illustrated walkthrough of the whole calculation, with the real bit-planes and an
interactive AND/popcount, is published at
<https://claude.ai/artifact/H5dLUnj3ow5bfgmieVznGi> (source: `intro.html`).

## Honest limits

- Only useful for small `k+l`. That happens to be the regime binary TE is
  actually used in (`k, l ≤ 3`), and the common `k=l=1` case is the best one,
  but the method does not scale in state space and should not pretend to.
- Plug-in and Miller–Madow only. No Bayesian or NSB estimator, no bootstrap
  confidence interval beyond the rotation null.
- The CPU paths are single-threaded. Tiles are independent, so
  `sync_parallelize` over tiles with per-thread count tables is the obvious
  next step; so is SIMD-widening the mask loops from `UInt64` to
  `SIMD[DType.uint64, 4]`.
- The GPU kernel is specialised to `k = l = 1` (any lag). Generalising means
  `2^(k+1)` mask registers and a `comptime`-unrolled source descent — the
  register budget is fine to about `k+l = 7`, which is the whole useful range,
  but it is not written yet.
- No GPU path for the lag scan or the rotation-surrogate null, which is the
  single most promising remaining target: both reuse the same resident planes
  at a different read offset, so a grid over (pair × lag × surrogate) would run
  thousands of independent counts over data that never leaves the device.
- `set_lag` requires the window reserved at construction via `umax`, since TE
  values are only comparable across lags on the same sample window.

## Ideas not pursued

- **Sparse mask representation.** Deep nodes pay a full `tw`-word pass however
  few bits are set. Switching to explicit bit indices once popcount drops below
  ~64 would cut deep levels from `2^d·nw` to ~`m` words. Narrow tiles already
  capture most of this for much less complexity.
- **Gray-code enumeration** of states, to reach each sibling by a single plane
  update. The DFS already shares prefixes, so the remaining gain is small.
- **Conditional / multivariate TE** — more conditioning planes, same tree, but
  the state space explodes right past the crossover.
