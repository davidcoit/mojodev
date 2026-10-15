# network.mojo -- all-pairs transfer entropy over a network of binary signals.
#
# For P signals there are P*(P-1) ordered pairs, and running the single-pair
# estimator P^2 times repeats a lot of work. The restructuring here is the one
# the plane order was already set up for.
#
# WHAT IS SHARED
# --------------
# In `BitTE` the state index is (y' << (k+l)) | (h << l) | g, and the planes
# are walked target-history first, source-history second. Of the k+l+1 planes,
# k+1 of them (the future plane and the k target-history planes) depend only on
# the *target*. So for one target, descend those first and materialise the
# 2^(k+1) masks they produce; every source then reuses them.
#
# Those 2^(k+1) masks are a *partition* of the usable time axis -- disjoint and
# covering -- because they enumerate every value of (y', h). That buys a second
# saving: under a given target mask M, the deepest source level needs only
#
#     c1 = popcount(M_partial & G)        c0 = |M_partial| - c1
#
# so the final complement mask is never materialised at all, only its count.
#
# COST, in plane passes over the time axis, per ordered pair:
#
#   per-pair BitTE     3 * 2^(k+l)
#   this               2^(k+1) * (3 * 2^(l-1) - 2)      + O(2^k) per target
#
#   k=l=1:  12 -> 4    (3.0x)        k=2,l=1: 24 -> 8   (3.0x)
#   k=1,l=2: 24 -> 16  (1.5x)        k=l=2:   48 -> 32  (1.5x)
#
# The win is largest when l is small, which is the usual binary-network case.
# Everything is tiled, so the target masks and the source mask stack stay in
# cache and the same decomposition maps onto a GPU block.

from std.bit import pop_count
from std.math import log2
from bitte import BitSeq, slice_into, f6


struct NetTE(Copyable, Movable):
    """All-pairs TE over P equal-length binary signals, one (k, l, u) setting."""

    var p: Int           # number of signals
    var n: Int           # samples per signal
    var k: Int
    var l: Int
    var u: Int
    var n0: Int
    var m: Int           # usable samples
    var nw: Int          # words per full plane
    var tw: Int          # tile width in words
    var nh: Int          # k + l
    var nstates: Int     # 1 << (nh + 1)
    var ntm: Int         # 1 << (k + 1), number of target masks

    var sigs: List[BitSeq]

    var tmask: List[UInt64]    # ntm masks, tw words each (current target/tile)
    var tm_pc: List[Int]       # popcount of each target mask in this tile
    var tstack: List[UInt64]   # (k+1) levels x tw, target descent scratch
    var tpl: List[UInt64]      # (k+1) target planes for this tile
    var spl: List[UInt64]      # l source planes for this tile
    var smask: List[UInt64]    # (l+1) levels x tw, source descent scratch

    var counts: List[Int]      # p * nstates: table per source, current target
    var te: List[Float64]      # p * p result matrix, te[src*p + tgt]

    var word_ops: Int

    def __init__(
        out self,
        sigs: List[BitSeq],
        k: Int,
        l: Int,
        u: Int,
        tile_words: Int = 64,
    ) raises:
        if len(sigs) < 2:
            raise Error("need at least two signals")
        if k < 1 or l < 1:
            raise Error("k and l must be >= 1")
        if u < 1:
            raise Error("lag u must be >= 1")
        self.p = len(sigs)
        self.n = sigs[0].n
        for i in range(self.p):
            if sigs[i].n != self.n:
                raise Error("all signals must have the same length")

        self.k = k
        self.l = l
        self.u = u
        self.nh = k + l
        self.nstates = 1 << (self.nh + 1)
        self.ntm = 1 << (k + 1)

        var need_target = k - 1
        var need_source = u + l - 2
        self.n0 = need_target if need_target > need_source else need_source
        self.m = self.n - 1 - self.n0
        if self.m < 1:
            raise Error("sequences too short for these k/l/u")
        self.nw = (self.m + 63) // 64
        self.tw = tile_words if tile_words < self.nw else self.nw
        if self.tw < 1:
            self.tw = 1

        self.sigs = sigs.copy()
        self.tmask = List[UInt64](length=self.ntm * self.tw, fill=0)
        self.tm_pc = List[Int](length=self.ntm, fill=0)
        self.tstack = List[UInt64](length=(k + 1) * self.tw, fill=0)
        self.tpl = List[UInt64](length=(k + 1) * self.tw, fill=0)
        self.spl = List[UInt64](length=l * self.tw, fill=0)
        self.smask = List[UInt64](length=(l + 1) * self.tw, fill=0)
        self.counts = List[Int](length=self.p * self.nstates, fill=0)
        self.te = List[Float64](length=self.p * self.p, fill=0.0)
        self.word_ops = 0

    # -- state index helpers --------------------------------------------------

    def base_index(self, tm: Int) -> Int:
        """Map a target-mask id (f << k) | h to its slot in the state index."""
        var f = tm >> self.k
        var h = tm & ((1 << self.k) - 1)
        return (f << self.nh) | (h << self.l)

    # -- the pass -------------------------------------------------------------

    def run(mut self):
        """Fill `te` for every ordered pair (source -> target)."""
        for i in range(len(self.te)):
            self.te[i] = 0.0
        self.word_ops = 0

        for tgt in range(self.p):
            for i in range(len(self.counts)):
                self.counts[i] = 0

            var t = 0
            while t < self.nw:
                var bits = self.m - (t << 6)
                if bits > (self.tw << 6):
                    bits = self.tw << 6
                var words = (bits + 63) // 64

                self._slice_target_planes(tgt, t, bits)
                self._build_target_masks(words, bits)

                for src in range(self.p):
                    if src == tgt:
                        continue
                    self._slice_source_planes(src, t, bits)
                    for tm in range(self.ntm):
                        if self.tm_pc[tm] == 0:
                            continue
                        self._source_root(
                            tm, self.base_index(tm), self.tm_pc[tm],
                            words, src * self.nstates,
                        )
                t += self.tw

            for src in range(self.p):
                if src == tgt:
                    continue
                self.te[src * self.p + tgt] = self._te_from(
                    src * self.nstates
                )

    def _slice_target_planes(mut self, tgt: Int, t: Int, bits: Int):
        """Plane 0 is the future bit; planes 1..k are the target history."""
        var off = t << 6
        slice_into(
            self.sigs[tgt], self.n0 + 1 + off, bits, self.tpl, 0
        )
        for j in range(self.k):
            slice_into(
                self.sigs[tgt], self.n0 - j + off, bits,
                self.tpl, (1 + j) * self.tw,
            )

    def _slice_source_planes(mut self, src: Int, t: Int, bits: Int):
        var off = t << 6
        for i in range(self.l):
            slice_into(
                self.sigs[src], self.n0 - self.u + 1 - i + off, bits,
                self.spl, i * self.tw,
            )

    def _build_target_masks(mut self, words: Int, bits: Int):
        """Materialise the 2^(k+1) masks over (y', h) for the current tile.

        They partition the tile, which is what lets the deepest source level
        skip its complement mask later.
        """
        var full = ~UInt64(0)
        for i in range(words):
            self.tstack[i] = full
        # The planes come out of slice_into with their tails already zeroed,
        # but this root mask is synthesised, so clamp it here. Without this the
        # complement children (root ^ child1) would pick up the junk tail bits.
        var rem = bits & 63
        if rem != 0:
            self.tstack[words - 1] = (UInt64(1) << UInt64(rem)) - 1
        self._tdfs(0, 0, bits, words)

    def _tdfs(mut self, d: Int, tm: Int, pc: Int, words: Int):
        var mo = d * self.tw
        var po = d * self.tw

        if d == self.k:
            # Deepest target level: children land directly in mask slots, and
            # the complement's count is a subtraction rather than a popcount.
            var tm1 = (tm << 1) | 1
            var tm0 = tm << 1
            var o1 = tm1 * self.tw
            var o0 = tm0 * self.tw
            var c1 = 0
            for i in range(words):
                var v = self.tstack[mo + i] & self.tpl[po + i]
                self.tmask[o1 + i] = v
                c1 += Int(pop_count(v))
            for i in range(words):
                self.tmask[o0 + i] = (
                    self.tstack[mo + i] ^ self.tmask[o1 + i]
                )
            self.tm_pc[tm1] = c1
            self.tm_pc[tm0] = pc - c1
            self.word_ops += 2 * words
            return

        var co = (d + 1) * self.tw
        var c1 = 0
        for i in range(words):
            var v = self.tstack[mo + i] & self.tpl[po + i]
            self.tstack[co + i] = v
            c1 += Int(pop_count(v))
        self.word_ops += words
        self._tdfs(d + 1, (tm << 1) | 1, c1, words)

        for i in range(words):
            self.tstack[co + i] = self.tstack[mo + i] ^ self.tstack[co + i]
        self.word_ops += words
        self._tdfs(d + 1, tm << 1, pc - c1, words)

    def _source_root(
        mut self, tm: Int, base: Int, pc: Int, words: Int, cbase: Int
    ):
        """Descend the l source planes from target mask `tm`.

        Level 0 reads `tmask` directly, so the target mask is never copied.
        """
        var mo = tm * self.tw

        if self.l == 1:
            var c1 = 0
            for i in range(words):
                c1 += Int(pop_count(self.tmask[mo + i] & self.spl[i]))
            self.counts[cbase + base + 1] += c1
            self.counts[cbase + base] += pc - c1
            self.word_ops += words
            return

        var co = self.tw   # level 1 of smask
        var c1 = 0
        for i in range(words):
            var v = self.tmask[mo + i] & self.spl[i]
            self.smask[co + i] = v
            c1 += Int(pop_count(v))
        self.word_ops += words
        if c1 != 0:
            self._sdfs(1, 1, c1, base, words, cbase)

        var c0 = pc - c1
        if c0 != 0:
            for i in range(words):
                self.smask[co + i] = (
                    self.tmask[mo + i] ^ self.smask[co + i]
                )
            self.word_ops += words
            self._sdfs(1, 0, c0, base, words, cbase)

    def _sdfs(
        mut self, d: Int, g: Int, pc: Int, base: Int, words: Int, cbase: Int
    ):
        var mo = d * self.tw
        var po = d * self.tw

        if d == self.l - 1:
            # Deepest source level: emit both counts, no child mask needed.
            var c1 = 0
            for i in range(words):
                c1 += Int(pop_count(self.smask[mo + i] & self.spl[po + i]))
            self.counts[cbase + base + (g | (1 << d))] += c1
            self.counts[cbase + base + g] += pc - c1
            self.word_ops += words
            return

        var co = (d + 1) * self.tw
        var c1 = 0
        for i in range(words):
            var v = self.smask[mo + i] & self.spl[po + i]
            self.smask[co + i] = v
            c1 += Int(pop_count(v))
        self.word_ops += words
        if c1 != 0:
            self._sdfs(d + 1, g | (1 << d), c1, base, words, cbase)

        var c0 = pc - c1
        if c0 != 0:
            for i in range(words):
                self.smask[co + i] = self.smask[mo + i] ^ self.smask[co + i]
            self.word_ops += words
            self._sdfs(d + 1, g, c0, base, words, cbase)

    # -- estimate -------------------------------------------------------------

    def _te_from(self, cbase: Int) -> Float64:
        var base = 1 << self.nh
        var hsize = 1 << self.k
        var n3 = List[Int](length=base, fill=0)
        var n1 = List[Int](length=2 * hsize, fill=0)
        var n2 = List[Int](length=hsize, fill=0)

        for s in range(self.nstates):
            var c = self.counts[cbase + s]
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
            var c = self.counts[cbase + s]
            if c == 0:
                continue
            var f = s >> self.nh
            var hg = s & (base - 1)
            var h = hg >> self.l
            var num = Float64(c) * Float64(n2[h])
            var den = Float64(n3[hg]) * Float64(n1[(f << self.k) | h])
            total += Float64(c) * log2(num / den)
        return total / Float64(self.m)

    def total_counted(self, src: Int) -> Int:
        var c = 0
        for s in range(self.nstates):
            c += self.counts[src * self.nstates + s]
        return c
