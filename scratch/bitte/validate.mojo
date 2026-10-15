# validate.mojo -- correctness of the bit-plane estimator.
#
# Two kinds of check:
#   1. agreement with the scalar reference walker, bit for bit, over a sweep of
#      (k, l, u) and sequence lengths that straddle the 64-bit word boundary;
#   2. agreement with closed-form transfer entropies on channels where the
#      answer is known analytically.

from std.math import log2
from bitte import BitSeq, BitTE, te_naive, te_naive_packed, f6, rotation_null
from gen import Rng, iid, iid_packed, channel, xor_channel2, autoregressive


def hb(p: Float64) -> Float64:
    """Binary entropy in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def unpack(s: BitSeq) -> List[Int]:
    var out = List[Int](length=s.n, fill=0)
    for i in range(s.n):
        out[i] = s.get(i)
    return out^


def check(label: String, got: Float64, want: Float64, tol: Float64) -> Int:
    var d = got - want
    var ad = -d if d < 0.0 else d
    if ad <= tol:
        print("  PASS ", label, " got", f6(got), " want", f6(want))
        return 0
    print("  FAIL ", label, " got", f6(got), " want", f6(want), " d", f6(ad))
    return 1


def main() raises:
    var fails = 0

    # ---------------------------------------------------------------------
    print("[1] bitwise vs scalar reference, sweeping k, l, u and length")
    print("    (lengths chosen to straddle the 64-bit word boundary)")
    var rng = Rng(20261015)
    var lengths = [63, 64, 65, 127, 128, 129, 1000, 4097]
    var worst = 0.0
    var cases = 0
    for li in range(len(lengths)):
        var n = lengths[li]
        var xs = iid(n, 0.5, rng)
        var ys = channel(xs, 2, 0.15, rng)
        for k in range(1, 4):
            for l in range(1, 4):
                for u in range(1, 4):
                    if n - 1 - (k - 1 if k - 1 > u + l - 2 else u + l - 2) < 8:
                        continue
                    var y = BitSeq(ys)
                    var x = BitSeq(xs)
                    var est = BitTE(y, x, k, l, u)
                    est.count()
                    # The count table must account for every usable sample.
                    if est.total_counted() != est.m:
                        print(
                            "  FAIL count conservation n",
                            n, "k", k, "l", l, "u", u,
                            "total", est.total_counted(), "m", est.m,
                        )
                        fails += 1
                    var a = est.te_bits()
                    var b = te_naive(ys, xs, k, l, u)
                    var d = a - b
                    var ad = -d if d < 0.0 else d
                    if ad > worst:
                        worst = ad

                    # The scalar walk over the packed arrays must agree too.
                    var c = te_naive_packed(y, x, k, l, u)
                    var d2 = a - c
                    var ad2 = -d2 if d2 < 0.0 else d2
                    if ad2 > worst:
                        worst = ad2

                    # Tiled counting must reproduce the streaming counts
                    # exactly, at every tile width including awkward ones.
                    var tiles = [1, 2, 3, 7, 64, 100000]
                    for ti in range(len(tiles)):
                        var before = est.n4.copy()
                        est.count_tiled(tiles[ti])
                        if est.total_counted() != est.m:
                            print("  FAIL tiled count conservation, tile",
                                  tiles[ti], "n", n, "k", k, "l", l, "u", u)
                            fails += 1
                        for s in range(est.nstates):
                            if est.n4[s] != before[s]:
                                print("  FAIL tiled != streaming, tile",
                                      tiles[ti], "n", n, "k", k, "l", l,
                                      "u", u, "state", s)
                                fails += 1
                                break
                    est.count()
                    cases += 1
    print("    ", cases, "configurations compared, worst |bitwise - scalar| =", worst)
    if worst > 1e-12:
        print("  FAIL reference agreement")
        fails += 1
    else:
        print("  PASS reference agreement (exact to double precision)")

    # ---------------------------------------------------------------------
    print("")
    print("[2] closed-form channels, n = 2,000,000")
    var n = 2000000

    # Perfect copy with lag 1: y[t] = x[t-1].  TE = 1 bit.
    var xp = iid_packed(n, rng)
    var xs2 = unpack(xp)
    var ys2 = channel(xs2, 1, 0.0, rng)
    var e1 = BitTE(BitSeq(ys2), xp, 1, 1, 1)
    e1.count()
    fails += check("copy lag 1, k=l=u=1   TE", e1.te_bits(), 1.0, 1e-4)

    # Binary symmetric channel, eps = 0.1: TE = 1 - H_b(eps).
    var ys3 = channel(xs2, 1, 0.1, rng)
    var e2 = BitTE(BitSeq(ys3), xp, 1, 1, 1)
    e2.count()
    fails += check("BSC eps=0.10          TE", e2.te_bits(), 1.0 - hb(0.1), 2e-3)

    # eps = 0.25.
    var ys4 = channel(xs2, 1, 0.25, rng)
    var e3 = BitTE(BitSeq(ys4), xp, 1, 1, 1)
    e3.count()
    fails += check("BSC eps=0.25          TE", e3.te_bits(), 1.0 - hb(0.25), 2e-3)

    # eps = 0.5 is a severed channel: TE = 0 up to estimator bias.
    var ys5 = channel(xs2, 1, 0.5, rng)
    var e4 = BitTE(BitSeq(ys5), xp, 1, 1, 1)
    e4.count()
    fails += check("severed eps=0.50      TE", e4.te_bits(), 0.0, 1e-4)
    print("       Miller-Madow on the severed channel:", f6(e4.te_bits_mm()))

    # Autoregressive target, unrelated source: predictable but not driven.
    var yar = autoregressive(n, 0.1, rng)
    var e5 = BitTE(BitSeq(yar), xp, 1, 1, 1)
    e5.count()
    fails += check("AR target, no input   TE", e5.te_bits(), 0.0, 1e-4)
    print("       (target is highly predictable from itself: TE still ~0)")

    # ---------------------------------------------------------------------
    print("")
    print("[3] lag argument: coupling planted at u = 3, scan u = 1..6")
    var ys6 = channel(xs2, 3, 0.05, rng)
    var best_u = 0
    var best_v = -1.0
    for u in range(1, 7):
        var e = BitTE(BitSeq(ys6), xp, 1, 1, u)
        e.count()
        var v = e.te_bits()
        var mark = ""
        if v > best_v:
            best_v = v
            best_u = u
        print("     u =", u, " TE =", f6(v), mark)
    print("     peak at u =", best_u, " expected 3,  1-H_b(0.05) =", f6(1.0 - hb(0.05)))
    if best_u != 3:
        print("  FAIL lag localisation")
        fails += 1
    else:
        print("  PASS lag localisation")

    # ---------------------------------------------------------------------
    print("")
    print("[4] source-history argument: y[t] = x[t-1] XOR x[t-2]")
    print("    either source bit alone is uninformative; the pair is decisive")
    var ys7 = xor_channel2(xs2, rng)
    var f1 = BitTE(BitSeq(ys7), xp, 1, 1, 1)
    f1.count()
    var f2 = BitTE(BitSeq(ys7), xp, 1, 2, 1)
    f2.count()
    var f3 = BitTE(BitSeq(ys7), xp, 1, 3, 1)
    f3.count()
    fails += check("l = 1                 TE", f1.te_bits(), 0.0, 1e-4)
    fails += check("l = 2                 TE", f2.te_bits(), 1.0, 1e-3)
    print("     l = 3                 TE =", f6(f3.te_bits()), " (no extra information)")

    # ---------------------------------------------------------------------
    print("")
    print("[5] rotation surrogates (null distribution, nearly free)")
    var xd = xp.doubled()
    var e6 = BitTE(BitSeq(ys4), xp, 2, 2, 1)   # real coupling, eps = 0.25
    e6.count()
    var obs = e6.te_bits()
    var ns = rotation_null(e6, xd, 24, obs)
    print("     real coupling   TE =", f6(obs))
    print("       surrogate mean", f6(ns.mean), " sd", f6(ns.sd), " p =", f6(ns.pval))
    print("       effective TE  =", f6(obs - ns.mean))

    var e7 = BitTE(BitSeq(yar), xp, 2, 2, 1)   # no coupling at all
    e7.count()
    var obs2 = e7.te_bits()
    var ns2 = rotation_null(e7, xd, 24, obs2)
    print("     AR target       TE =", f6(obs2))
    print("       surrogate mean", f6(ns2.mean), " sd", f6(ns2.sd), " p =", f6(ns2.pval))
    print("       effective TE  =", f6(obs2 - ns2.mean))
    if ns.pval > 0.05:
        print("  FAIL coupled pair should be significant")
        fails += 1
    if ns2.pval < 0.05:
        print("  FAIL uncoupled pair should not be significant")
        fails += 1
    if ns.pval <= 0.05 and ns2.pval >= 0.05:
        print("  PASS surrogate test separates coupled from uncoupled")

    print("")
    if fails == 0:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES:", fails)
