# bitte.mojo -- Bitwise Transfer Entropy for binary sequences.
#
# Transfer entropy from source X to target Y, with target history k, source
# history l and source lag u:
#
#   TE = sum p(y', h, g) * log2[ p(y'|h,g) / p(y'|h) ]
#
#   y' = y[n+1]                            (the future target bit)
#   h  = y[n], y[n-1], ... y[n-k+1]        (k target history bits)
#   g  = x[n-u+1], ... x[n-u-l+2]          (l source history bits)
#
# Everything reduces to one count table n4[y', h, g] over 2^(1+k+l) bins; the
# three lower-order marginals fall out of it by summation.
#
# THE IDEA
# --------
# The usual estimator walks the time axis once and builds an integer index per
# sample.  Here the time axis never appears in the inner loop.  A binary
# sequence is bit-packed, and "the j-th history bit over all n" is just that
# packed array read at a shifted bit offset -- a *bit-plane*.  The set of time
# steps matching a given state is then
#
#   match = P_0^(b0) & P_1^(b1) & ... & P_d^(bd)      where P^(1) = P, P^(0) = ~P
#
# and its count is popcount(match).  64 time steps are tested per instruction.
#
# Two refinements make it cheap:
#
#  1. Share prefixes.  Enumerating the 2^(k+l) states as a DFS over the planes
#     means each node does one AND pass over the partial mask, not (k+l) of
#     them, so the per-state cost collapses from (k+l) passes to ~2.
#  2. Complement for free.  The zero-child's mask is  mask ^ mask1, because
#     mask & ~P = mask ^ (mask & P).  So the second child costs one XOR pass
#     and *no* popcount at all: its count is parent_count - count1.
#
#  Plus: a node whose mask is empty has an all-zero subtree, so it is pruned.
#
# Cost: ~3 * 2^(k+l) * ceil(m/64) word ops, versus ~m*(k+l) for the scalar
# walk.  Bitwise wins while 2^(k+l) < ~20*(k+l), i.e. up to k+l ~ 8.

from std.bit import pop_count
from std.math import log2

comptime LN2: Float64 = 0.6931471805599453


# ---------------------------------------------------------------- BitSeq -----

struct BitSeq(Copyable, Movable):
    """A binary sequence packed LSB-first into 64-bit words.

    One guard word of zeros is always allocated past the end so that the
    two-word funnel read in `slice_into` never runs off the buffer.
    """

    var w: List[UInt64]
    var n: Int

    def __init__(out self, n: Int):
        self.n = n
        self.w = List[UInt64](length=(n + 63) // 64 + 1, fill=0)

    def __init__(out self, bits: List[Int]):
        self = BitSeq(len(bits))
        for i in range(len(bits)):
            if bits[i] != 0:
                self.w[i >> 6] |= UInt64(1) << UInt64(i & 63)

    def get(self, i: Int) -> Int:
        return Int((self.w[i >> 6] >> UInt64(i & 63)) & 1)

    def set(mut self, i: Int, b: Int):
        var mask = UInt64(1) << UInt64(i & 63)
        if b != 0:
            self.w[i >> 6] |= mask
        else:
            self.w[i >> 6] &= ~mask

    def doubled(self) -> BitSeq:
        """Return the sequence concatenated with itself.

        A cyclic rotation of the original is then an ordinary linear bit-slice
        of this, which is what makes rotation surrogates nearly free.
        """
        var d = BitSeq(2 * self.n)
        for i in range(self.n):
            if self.get(i) != 0:
                d.set(i, 1)
                d.set(i + self.n, 1)
        return d^


def slice_into(src: BitSeq, start: Int, m: Int, mut dst: List[UInt64], off: Int):
    """Extract bits [start, start+m) of `src`, repacked to bit 0, into dst[off:].

    This is the whole trick behind a bit-plane: a word-wise funnel shift.  The
    `(hi << 1) << (63 - r)` form keeps both shift counts below 64 so that the
    r == 0 case lands on zero instead of undefined behaviour.
    """
    var words = (m + 63) // 64
    for i in range(words):
        var bp = start + (i << 6)
        var q = bp >> 6
        var r = UInt64(bp & 63)
        dst[off + i] = (src.w[q] >> r) | ((src.w[q + 1] << 1) << (63 - r))
    var rem = m & 63
    if rem != 0:
        # Zero the unused high bits of the last word so popcounts stay honest.
        dst[off + words - 1] &= (UInt64(1) << UInt64(rem)) - 1


# ----------------------------------------------------------------- BitTE -----

struct BitTE(Copyable, Movable):
    """Bit-plane transfer entropy estimator for one (k, l, u) configuration."""

    var k: Int          # target history length
    var l: Int          # source history length
    var u: Int          # source lag (u >= 1)
    var umax: Int       # largest lag this instance's sample window allows
    var xref: BitSeq    # the source, kept so set_lag can re-slice it
    var n: Int          # sequence length
    var n0: Int         # first usable time index
    var m: Int          # number of usable samples
    var nw: Int         # words per plane
    var nh: Int         # k + l, number of conditioning planes
    var nstates: Int    # 1 << (nh + 1)
    var planes: List[UInt64]   # nh planes, flat, plane d at [d*nw, (d+1)*nw)
    var bitpos: List[Int]      # state-index bit position for each plane
    var fpl: List[UInt64]      # the future-target plane
    var masks: List[UInt64]    # nh+1 DFS levels, same flat layout
    var tmasks: List[UInt64]   # nh+1 DFS levels, one tile wide (count_tiled)
    var tstride: Int           # tile width in words
    var n4: List[Int]          # joint counts, index (y'<<nh) | (h<<l) | g
    var nodes: Int             # DFS nodes expanded (pruning telemetry)
    var word_ops: Int          # 64-bit word operations in the count pass
    var tree_nodes: Int        # nodes an unpruned walk would have expanded

    def __init__(
        out self, y: BitSeq, x: BitSeq, k: Int, l: Int, u: Int, umax: Int = 0
    ) raises:
        """`umax` reserves the sample window for a later `set_lag` up to umax.

        TE values are only comparable across lags if they are estimated on the
        same sample window, so a lag scan should pass the largest lag it will
        ever ask for.
        """
        if y.n != x.n:
            raise Error("source and target must be the same length")
        if k < 1 or l < 1:
            raise Error("k and l must be >= 1")
        if u < 1:
            raise Error("lag u must be >= 1")
        if k + l > 24:
            raise Error("k + l > 24 would need a count table over 32M bins")

        self.k = k
        self.l = l
        self.u = u
        self.umax = umax if umax > u else u
        self.xref = x.copy()
        self.n = y.n
        self.nh = k + l
        self.nstates = 1 << (self.nh + 1)

        # Earliest n for which every bit the formula needs exists; the latest
        # is n-2, because the future bit is y[n+1].
        var need_target = k - 1
        var need_source = self.umax + l - 2
        self.n0 = need_target if need_target > need_source else need_source
        self.m = self.n - 1 - self.n0
        if self.m < 1:
            raise Error("sequence too short for these k/l/u")
        self.nw = (self.m + 63) // 64

        self.planes = List[UInt64](length=self.nh * self.nw, fill=0)
        self.fpl = List[UInt64](length=self.nw, fill=0)
        self.masks = List[UInt64](length=(self.nh + 1) * self.nw, fill=0)
        self.tmasks = List[UInt64]()
        self.tstride = 0
        self.n4 = List[Int](length=self.nstates, fill=0)
        self.bitpos = List[Int](length=self.nh, fill=0)
        self.nodes = 0
        self.word_ops = 0
        self.tree_nodes = 0

        # Future plane: y[n0+p+1] for p in [0, m).
        slice_into(y, self.n0 + 1, self.m, self.fpl, 0)

        # Target history planes: y[n0+p-j], state bit l+j.
        for j in range(k):
            slice_into(y, self.n0 - j, self.m, self.planes, j * self.nw)
            self.bitpos[j] = l + j

        # Source history planes: x[n0+p-u+1-i], state bit i.
        for i in range(l):
            slice_into(
                x, self.n0 - u + 1 - i, self.m, self.planes, (k + i) * self.nw
            )
            self.bitpos[k + i] = i

    def set_source_rotation(mut self, xd: BitSeq, r: Int):
        """Re-read the source planes from the doubled source, rotated by r.

        A cyclic rotation destroys any genuine source->target timing while
        preserving the source's own autocorrelation exactly, which makes it a
        well-behaved surrogate for a null distribution.  Here it costs only a
        re-slice of l planes -- no repacking, no shuffling.
        """
        for i in range(self.l):
            slice_into(
                xd,
                self.n0 - self.u + 1 - i + r,
                self.m,
                self.planes,
                (self.k + i) * self.nw,
            )

    def count(mut self):
        """Fill n4 by walking the AND-tree over all 2^(k+l) history states."""
        for s in range(self.nstates):
            self.n4[s] = 0
        self.nodes = 0
        self.word_ops = 0
        self.tree_nodes = (1 << self.nh) - 1

        # Root mask: every usable time step, with the tail of the last word
        # zeroed.  Every descendant is an AND of this, so the tail stays clean.
        var full = ~UInt64(0)
        for i in range(self.nw):
            self.masks[i] = full
        var rem = self.m & 63
        if rem != 0:
            self.masks[self.nw - 1] = (UInt64(1) << UInt64(rem)) - 1

        self._dfs(0, 0, self.m)

    def _dfs(mut self, d: Int, s: Int, pc: Int):
        var mo = d * self.nw

        if d == self.nh:
            # Leaf: one AND+popcount pass splits the matching time steps by the
            # future bit.  The complement count needs no work at all.
            var c1 = 0
            for i in range(self.nw):
                c1 += Int(pop_count(self.masks[mo + i] & self.fpl[i]))
            self.n4[s] = pc - c1
            self.n4[s | (self.nstates >> 1)] = c1
            self.word_ops += self.nw
            return

        var co = (d + 1) * self.nw
        var po = d * self.nw
        var bit = 1 << self.bitpos[d]
        self.nodes += 1

        # Child with this plane's bit = 1: mask & plane, counted as we go.
        var c1 = 0
        for i in range(self.nw):
            var v = self.masks[mo + i] & self.planes[po + i]
            self.masks[co + i] = v
            c1 += Int(pop_count(v))
        self.word_ops += self.nw
        if c1 != 0:
            self._dfs(d + 1, s | bit, c1)

        # Child with bit = 0.  The subtree above only ever wrote levels
        # >= d+2, so masks[co:] still holds mask1 and
        #   mask & ~plane == mask ^ (mask & plane) == mask ^ mask1.
        # One XOR pass, and the count is already known.
        var c0 = pc - c1
        if c0 != 0:
            for i in range(self.nw):
                self.masks[co + i] = self.masks[mo + i] ^ self.masks[co + i]
            self.word_ops += self.nw
            self._dfs(d + 1, s, c0)

    def set_lag(mut self, u: Int) raises:
        """Re-point the source planes at a new lag, reusing everything else.

        A lag scan is the common real task, and the target planes, the future
        plane and all the buffers are lag-independent -- only the l source
        planes move.  Rebuilding the whole estimator per lag throws that away.
        """
        if u < 1 or u > self.umax:
            raise Error("set_lag needs 1 <= u <= umax given at construction")
        self.u = u
        for i in range(self.l):
            slice_into(
                self.xref,
                self.n0 - u + 1 - i,
                self.m,
                self.planes,
                (self.k + i) * self.nw,
            )

    # -- tiled counting -------------------------------------------------------

    def count_tiled(mut self, tile_words: Int):
        """Same counts as `count`, but blocked along the time axis.

        `count` walks the whole time axis once per tree node, so the mask
        buffers are 8*nw bytes per level and the deep levels stream gigabytes
        through RAM.  Blocking inverts the loops: for one tile of the time
        axis, run the *entire* 2^(k+l) tree, then move on.  Every mask level
        is then only 8*tile_words bytes, so the working set is L1-resident and
        each plane is read from RAM once per tile instead of 2^d times.
        """
        for s in range(self.nstates):
            self.n4[s] = 0
        self.nodes = 0
        self.word_ops = 0

        var tw_max = tile_words if tile_words < self.nw else self.nw
        if tw_max < 1:
            tw_max = 1
        self.tree_nodes = ((1 << self.nh) - 1) * (
            (self.nw + tw_max - 1) // tw_max
        )
        if self.tstride != tw_max:
            self.tmasks = List[UInt64](length=(self.nh + 1) * tw_max, fill=0)
            self.tstride = tw_max

        var rem = self.m & 63
        var full = ~UInt64(0)
        var t = 0
        while t < self.nw:
            var tw = tw_max
            if t + tw > self.nw:
                tw = self.nw - t
            var pc = 0
            for i in range(tw):
                var v = full
                if rem != 0 and (t + i) == self.nw - 1:
                    v = (UInt64(1) << UInt64(rem)) - 1
                self.tmasks[i] = v
                pc += Int(pop_count(v))
            if pc != 0:
                self._dfs_tile(0, 0, pc, t, tw)
            t += tw_max

    def count_auto(mut self):
        """Count with whichever strategy measures fastest for this k+l.

        Measured here (1M observations, dense fair-coin data): streaming wins
        up to k+l = 7, where the mask levels still fit in cache and per-tile
        overhead is not yet amortised.  Past that, the narrowest tiles win by a
        growing margin -- and not mainly for cache reasons.  A one-word tile is
        64 consecutive samples, which can contain at most 64 of the 2^(k+l)
        history states, so nearly every subtree is empty and gets pruned.  At
        k+l = 14 that takes the live node count to ~3% of the tree and the run
        time from 1905 ms to 101 ms.
        """
        if self.nh <= 7:
            self.count()
        elif self.nh == 8:
            self.count_tiled(2)
        else:
            self.count_tiled(1)

    def _dfs_tile(mut self, d: Int, s: Int, pc: Int, t: Int, tw: Int):
        var mo = d * self.tstride

        if d == self.nh:
            var c1 = 0
            for i in range(tw):
                c1 += Int(pop_count(self.tmasks[mo + i] & self.fpl[t + i]))
            self.n4[s] += pc - c1
            self.n4[s | (self.nstates >> 1)] += c1
            self.word_ops += tw
            return

        var co = (d + 1) * self.tstride
        var po = d * self.nw + t
        var bit = 1 << self.bitpos[d]
        self.nodes += 1

        var c1 = 0
        for i in range(tw):
            var v = self.tmasks[mo + i] & self.planes[po + i]
            self.tmasks[co + i] = v
            c1 += Int(pop_count(v))
        self.word_ops += tw
        if c1 != 0:
            self._dfs_tile(d + 1, s | bit, c1, t, tw)

        var c0 = pc - c1
        if c0 != 0:
            for i in range(tw):
                self.tmasks[co + i] = self.tmasks[mo + i] ^ self.tmasks[co + i]
            self.word_ops += tw
            self._dfs_tile(d + 1, s, c0, t, tw)

    # -- estimates ------------------------------------------------------------

    def te_bits(self) -> Float64:
        return self._te(False)

    def te_bits_mm(self) -> Float64:
        """Plug-in estimate with the Miller-Madow bias correction applied."""
        return self._te(True)

    def _te(self, miller_madow: Bool) -> Float64:
        var base = 1 << self.nh
        var hsize = 1 << self.k

        var n3 = List[Int](length=base, fill=0)          # (h, g)
        var n1 = List[Int](length=2 * hsize, fill=0)     # (y', h)
        var n2 = List[Int](length=hsize, fill=0)         # (h)

        for s in range(self.nstates):
            var c = self.n4[s]
            if c == 0:
                continue
            var f = s >> self.nh
            var hg = s & (base - 1)
            var h = hg >> self.l
            n3[hg] += c
            n1[(f << self.k) | h] += c
            n2[h] += c

        var total = 0.0
        for s in range(self.nstates):
            var c = self.n4[s]
            if c == 0:
                continue
            var f = s >> self.nh
            var hg = s & (base - 1)
            var h = hg >> self.l
            var num = Float64(c) * Float64(n2[h])
            var den = Float64(n3[hg]) * Float64(n1[(f << self.k) | h])
            total += Float64(c) * log2(num / den)
        var te = total / Float64(self.m)

        if miller_madow:
            # TE = H(y',h) - H(h) - H(y',h,g) + H(h,g); each plug-in entropy
            # carries a -(K-1)/(2m) bias, K = occupied bins.
            var k4 = 0
            var k3 = 0
            var k1 = 0
            var k2 = 0
            for s in range(self.nstates):
                if self.n4[s] != 0:
                    k4 += 1
            for i in range(base):
                if n3[i] != 0:
                    k3 += 1
            for i in range(2 * hsize):
                if n1[i] != 0:
                    k1 += 1
            for i in range(hsize):
                if n2[i] != 0:
                    k2 += 1
            te += Float64(k1 - k2 - k4 + k3) / (2.0 * Float64(self.m) * LN2)
        return te

    def occupied(self) -> Int:
        var c = 0
        for s in range(self.nstates):
            if self.n4[s] != 0:
                c += 1
        return c

    def total_counted(self) -> Int:
        var c = 0
        for s in range(self.nstates):
            c += self.n4[s]
        return c


# ---------------------------------------------------- rotation surrogates ----

struct NullStats(Copyable, Movable):
    var mean: Float64
    var sd: Float64
    var pval: Float64
    var nsurr: Int

    def __init__(out self, mean: Float64, sd: Float64, pval: Float64, nsurr: Int):
        self.mean = mean
        self.sd = sd
        self.pval = pval
        self.nsurr = nsurr


def rotation_null(
    mut est: BitTE, xd: BitSeq, nsurr: Int, observed: Float64
) -> NullStats:
    """Null distribution of TE under cyclic rotations of the source.

    Rotating the source leaves its marginal and its entire autocorrelation
    untouched while destroying its temporal alignment with the target, so the
    spread of these values is the bias floor of the estimator for *this* data
    rather than for an idealised independent pair.  In bit-packed form a
    rotation is just a different read offset, so a surrogate costs one
    re-slice plus one count pass -- no reshuffling of samples at all.
    """
    var vals = List[Float64](length=nsurr, fill=0.0)
    var step = est.n // (nsurr + 1)
    if step < 1:
        step = 1
    var sum = 0.0
    var nge = 0
    for i in range(nsurr):
        var r = ((i + 1) * step) % est.n
        est.set_source_rotation(xd, r)
        est.count()
        var v = est.te_bits()
        vals[i] = v
        sum += v
        if v >= observed:
            nge += 1
    est.set_source_rotation(xd, 0)   # put the real source back
    est.count()

    var mean = sum / Float64(nsurr)
    var var_ = 0.0
    for i in range(nsurr):
        var d = vals[i] - mean
        var_ += d * d
    var sd = (var_ / Float64(nsurr)) ** 0.5
    var pval = Float64(nge + 1) / Float64(nsurr + 1)
    return NullStats(mean, sd, pval, nsurr)


# ------------------------------------------------------- reference walker ----

def te_naive(y: List[Int], x: List[Int], k: Int, l: Int, u: Int) -> Float64:
    """Textbook scalar estimator: one pass over time, index built per sample.

    Kept as the correctness oracle and the speed baseline.
    """
    var n = len(y)
    var nh = k + l
    var nstates = 1 << (nh + 1)
    var need_target = k - 1
    var need_source = u + l - 2
    var n0 = need_target if need_target > need_source else need_source
    var m = n - 1 - n0
    var n4 = List[Int](length=nstates, fill=0)

    for p in range(m):
        var t = n0 + p
        var idx = 0
        for j in range(k):
            if y[t - j] != 0:
                idx |= 1 << (l + j)
        for i in range(l):
            if x[t - u + 1 - i] != 0:
                idx |= 1 << i
        if y[t + 1] != 0:
            idx |= 1 << nh
        n4[idx] += 1

    var base = 1 << nh
    var hsize = 1 << k
    var n3 = List[Int](length=base, fill=0)
    var n1 = List[Int](length=2 * hsize, fill=0)
    var n2 = List[Int](length=hsize, fill=0)
    for s in range(nstates):
        var c = n4[s]
        if c == 0:
            continue
        var f = s >> nh
        var hg = s & (base - 1)
        var h = hg >> l
        n3[hg] += c
        n1[(f << k) | h] += c
        n2[h] += c

    var total = 0.0
    for s in range(nstates):
        var c = n4[s]
        if c == 0:
            continue
        var f = s >> nh
        var hg = s & (base - 1)
        var h = hg >> l
        var num = Float64(c) * Float64(n2[h])
        var den = Float64(n3[hg]) * Float64(n1[(f << k) | h])
        total += Float64(c) * log2(num / den)
    return total / Float64(m)


def te_naive_packed(y: BitSeq, x: BitSeq, k: Int, l: Int, u: Int) -> Float64:
    """Scalar walk, but reading the same bit-packed arrays the AND-tree uses.

    `te_naive` reads List[Int] -- 8 bytes per observation -- so comparing
    against it conflates two separate wins: the 64x smaller footprint and the
    AND-tree itself.  This baseline keeps the tight representation and changes
    only the algorithm, so the difference is attributable to the algorithm.
    """
    var nh = k + l
    var nstates = 1 << (nh + 1)
    var need_target = k - 1
    var need_source = u + l - 2
    var n0 = need_target if need_target > need_source else need_source
    var m = y.n - 1 - n0
    var n4 = List[Int](length=nstates, fill=0)

    for p in range(m):
        var t = n0 + p
        var idx = 0
        for j in range(k):
            idx |= y.get(t - j) << (l + j)
        for i in range(l):
            idx |= x.get(t - u + 1 - i) << i
        idx |= y.get(t + 1) << nh
        n4[idx] += 1

    var base = 1 << nh
    var hsize = 1 << k
    var n3 = List[Int](length=base, fill=0)
    var n1 = List[Int](length=2 * hsize, fill=0)
    var n2 = List[Int](length=hsize, fill=0)
    for s in range(nstates):
        var c = n4[s]
        if c == 0:
            continue
        var f = s >> nh
        var hg = s & (base - 1)
        var h = hg >> l
        n3[hg] += c
        n1[(f << k) | h] += c
        n2[h] += c

    var total = 0.0
    for s in range(nstates):
        var c = n4[s]
        if c == 0:
            continue
        var f = s >> nh
        var hg = s & (base - 1)
        var h = hg >> l
        var num = Float64(c) * Float64(n2[h])
        var den = Float64(n3[hg]) * Float64(n1[(f << k) | h])
        total += Float64(c) * log2(num / den)
    return total / Float64(m)


# ------------------------------------------------------------- printing ------

def f6(x: Float64) -> String:
    """Fixed six-decimal formatting; Mojo's default float print is too wide."""
    var neg = x < 0.0
    var v = -x if neg else x
    var scaled = Int(v * 1000000.0 + 0.5)
    var ip = scaled // 1000000
    var fp = scaled % 1000000
    var fps = String(fp)
    while fps.byte_length() < 6:
        fps = "0" + fps
    var s = String(ip) + "." + fps
    return "-" + s if neg else s
