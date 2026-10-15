# bitte — bitwise transfer entropy for binary sequences

Pure Mojo (1.1.0), no `max`, no GPU, no Python. An experiment in estimating
transfer entropy between two binary time series using bit-packed storage and
almost nothing but AND / XOR / NOT / shift / popcount.

```
bitte.mojo      the estimator
gen.mojo        xorshift64* and the synthetic channels used for testing
validate.mojo   correctness: vs a scalar oracle, and vs closed-form answers
bench.mojo      cost measurements
smoke.mojo      smallest end-to-end check

mojo build -O3 validate.mojo -o validate && ./validate
mojo build -O3 bench.mojo    -o bench    && ./bench
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

## Honest limits

- Only useful for small `k+l`. That happens to be the regime binary TE is
  actually used in (`k, l ≤ 3`), and the common `k=l=1` case is the best one,
  but the method does not scale in state space and should not pretend to.
- Plug-in and Miller–Madow only. No Bayesian or NSB estimator, no bootstrap
  confidence interval beyond the rotation null.
- Single-threaded. Tiles are independent, so `sync_parallelize` over tiles with
  per-thread count tables is the obvious next step; so is SIMD-widening the
  mask loops from `UInt64` to `SIMD[DType.uint64, 4]`.
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
