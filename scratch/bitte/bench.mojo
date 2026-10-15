# bench.mojo -- where the bit-plane form wins, and where it stops winning.
#
# Four estimators over the same data:
#
#   scalar/list   textbook walk over List[Int]      8 bytes per observation
#   scalar/packed textbook walk over BitSeq         1 bit per observation
#   AND-tree      count()        streaming          1 bit, masks sized 8*nw
#   AND-tree/tile count_tiled()  time-blocked       1 bit, masks L1-resident
#
# scalar/packed is the honest baseline for the algorithm: it holds the tight
# representation fixed and changes only how the counting is done.
#
# Cost models:
#   scalar   ~ m * (k + l)                 ops, flat in the state count
#   AND-tree ~ 3 * 2^(k+l) * m/64          word ops, flat in (k+l) per sample
#
# Every timing is the best of REPS runs.

from std.time import perf_counter_ns
from bitte import BitSeq, BitTE, te_naive, te_naive_packed, f6
from gen import Rng, iid, iid_packed, channel

comptime REPS: Int = 3


def ms(ns: Int) -> String:
    return f6(Float64(ns) / 1000000.0)


def pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out = out + " "
    return out^


def padl(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out = " " + out
    return out^


def main() raises:
    var rng = Rng(777)
    var n = 1000000

    print("n =", n, "observations.  Packed:", n // 8 // 1024,
          "KB per sequence;  List[Int]:", n * 8 // 1024 // 1024, "MB.")
    print("All timings are best of", REPS, "runs.")
    print("")

    var xs = iid(n, 0.5, rng)
    var ys = channel(xs, 2, 0.12, rng)
    var y = BitSeq(ys)
    var x = BitSeq(xs)

    # ------------------------------------------------------------------ A ---
    print("[A] dense fair-coin data, lag 2, k and l swept")
    print(pad("  k+l", 7), padl("stream ms", 11), padl("auto ms", 11),
          padl("sc/pack ms", 12), padl("sc/list ms", 12),
          padl("auto vs", 10), padl("auto vs", 10))
    print(pad("", 7), padl("", 11), padl("", 11), padl("", 12), padl("", 12),
          padl("sc/pack", 10), padl("sc/list", 10))
    var ks = [1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]
    var ls = [1, 2, 2, 3, 3, 3, 4, 4, 4, 6, 7]
    for i in range(len(ks)):
        var k = ks[i]
        var l = ls[i]
        var est = BitTE(y, x, k, l, 2)

        var t_stream = 0
        var t_tiled = 0
        var t_pack = 0
        var t_list = 0
        for r in range(REPS):
            var a0 = perf_counter_ns()
            est.count()
            var v1 = est.te_bits()
            var a1 = perf_counter_ns()
            if r == 0 or Int(a1 - a0) < t_stream:
                t_stream = Int(a1 - a0)

            var b0 = perf_counter_ns()
            est.count_auto()
            var v2 = est.te_bits()
            var b1 = perf_counter_ns()
            if r == 0 or Int(b1 - b0) < t_tiled:
                t_tiled = Int(b1 - b0)

            var c0 = perf_counter_ns()
            var v3 = te_naive_packed(y, x, k, l, 2)
            var c1 = perf_counter_ns()
            if r == 0 or Int(c1 - c0) < t_pack:
                t_pack = Int(c1 - c0)

            var d0 = perf_counter_ns()
            var v4 = te_naive(ys, xs, k, l, 2)
            var d1 = perf_counter_ns()
            if r == 0 or Int(d1 - d0) < t_list:
                t_list = Int(d1 - d0)

            if v1 != v2 or v1 != v3 or v1 != v4:
                print("    MISMATCH", f6(v1), f6(v2), f6(v3), f6(v4))

        print(pad(String(k) + "+" + String(l), 7),
              padl(ms(t_stream), 11), padl(ms(t_tiled), 11),
              padl(ms(t_pack), 12), padl(ms(t_list), 12),
              padl(f6(Float64(t_pack) / Float64(t_tiled)) + "x", 10),
              padl(f6(Float64(t_list) / Float64(t_tiled)) + "x", 10))

    # ------------------------------------------------------------------ B ---
    print("")
    print("[B] tile width sweep (tile = words of time axis held in registers")
    print("    and L1; nw =", (n - 1) // 64 + 1, "words is one whole pass)")
    print("    ms, and the live DFS nodes as a fraction of the full tree")
    var tiles = [1, 2, 4, 8, 16, 64, 256, 2048, 1000000]
    var cfg_k = [2, 3, 4, 5, 6, 7]
    for ci in range(len(cfg_k)):
        var k = cfg_k[ci]
        var l = cfg_k[ci]
        var est = BitTE(y, x, k, l, 2)
        var l1 = "  k+l=" + padl(String(k + l), 2) + "  ms   "
        var l2 = "          live "
        for ti in range(len(tiles)):
            var best = 0
            for r in range(REPS):
                var a0 = perf_counter_ns()
                est.count_tiled(tiles[ti])
                var a1 = perf_counter_ns()
                if r == 0 or Int(a1 - a0) < best:
                    best = Int(a1 - a0)
            var pct = 100.0 * Float64(est.nodes) / Float64(est.tree_nodes)
            l1 = l1 + padl(ms(best), 11)
            l2 = l2 + padl(f6(pct) + "%", 11)
        print(l1)
        print(l2)
    var hdr = "          tile "
    for ti in range(len(tiles)):
        hdr = hdr + padl(String(tiles[ti]), 11)
    print(hdr)
    print("    (last column is tile >= nw, i.e. one unblocked pass)")

    # ------------------------------------------------------------------ C ---
    print("")
    print("[C] sparse data (p = 0.02): empty masks let the DFS prune subtrees")
    var xsp = iid(n, 0.02, rng)
    var ysp = channel(xsp, 1, 0.02, rng)
    var ybs = BitSeq(ysp)
    var xbs = BitSeq(xsp)
    print(pad("  k+l", 7), padl("auto ms", 11), padl("sc/pack ms", 12),
          padl("speedup", 10), padl("live nodes", 14))
    var kk = [4, 5, 6, 7, 8]
    for i in range(len(kk)):
        var k = kk[i]
        var est = BitTE(ybs, xbs, k, k, 1)
        var tb = 0
        var tp = 0
        for r in range(REPS):
            var a0 = perf_counter_ns()
            est.count_auto()
            var v1 = est.te_bits()
            var a1 = perf_counter_ns()
            if r == 0 or Int(a1 - a0) < tb:
                tb = Int(a1 - a0)
            var b0 = perf_counter_ns()
            var v2 = te_naive_packed(ybs, xbs, k, k, 1)
            var b1 = perf_counter_ns()
            if r == 0 or Int(b1 - b0) < tp:
                tp = Int(b1 - b0)
            if v1 != v2:
                print("    MISMATCH", f6(v1), f6(v2))
        var pct = 100.0 * Float64(est.nodes) / Float64(est.tree_nodes)
        print(pad(String(2 * k), 7), padl(ms(tb), 11), padl(ms(tp), 12),
              padl(f6(Float64(tp) / Float64(tb)) + "x", 10),
              padl(f6(pct) + "%", 14))

    # ------------------------------------------------------------------ D ---
    print("")
    print("[D] length scaling at k = l = 2, lag 1")
    print(pad("  n", 11), padl("auto ms", 11), padl("sc/pack ms", 12),
          padl("speedup", 10), padl("ns/obs", 10))
    var sizes = [10000, 100000, 1000000, 8000000, 32000000]
    for i in range(len(sizes)):
        var sz = sizes[i]
        var xp = iid_packed(sz, rng)
        var yi = List[Int](length=sz, fill=0)
        for t in range(sz):
            yi[t] = xp.get(t - 1) ^ rng.bit(0.12) if t >= 1 else rng.bit(0.5)
        var yb = BitSeq(yi)
        var est = BitTE(yb, xp, 2, 2, 1)
        var tb = 0
        var tp = 0
        for r in range(REPS):
            var a0 = perf_counter_ns()
            est.count_auto()
            var v1 = est.te_bits()
            var a1 = perf_counter_ns()
            if r == 0 or Int(a1 - a0) < tb:
                tb = Int(a1 - a0)
            var b0 = perf_counter_ns()
            var v2 = te_naive_packed(yb, xp, 2, 2, 1)
            var b1 = perf_counter_ns()
            if r == 0 or Int(b1 - b0) < tp:
                tp = Int(b1 - b0)
            if v1 != v2:
                print("    MISMATCH", f6(v1), f6(v2))
        print(pad(String(sz), 11), padl(ms(tb), 11), padl(ms(tp), 12),
              padl(f6(Float64(tp) / Float64(tb)) + "x", 10),
              padl(f6(Float64(tb) / Float64(sz)), 10))

    # ------------------------------------------------------------------ E ---
    print("")
    print("[E] lag scan u = 1..32 at k = l = 2 (the usual real task)")
    var umax = 32

    # One estimator, re-pointed at each lag: planes for the target and the
    # future are built once and every lag reuses them.
    var t_scan = 0
    var peak_u = 0
    var peak_v = -1.0
    for r in range(REPS):
        var est = BitTE(y, x, 2, 2, 1, umax)
        var a0 = perf_counter_ns()
        for u in range(1, umax + 1):
            est.set_lag(u)
            est.count_auto()
            var v = est.te_bits()
            if v > peak_v:
                peak_v = v
                peak_u = u
        var a1 = perf_counter_ns()
        if r == 0 or Int(a1 - a0) < t_scan:
            t_scan = Int(a1 - a0)

    # Same scan, but rebuilding the estimator from scratch per lag.
    var t_rebuild = 0
    for r in range(REPS):
        var a0 = perf_counter_ns()
        for u in range(1, umax + 1):
            var e = BitTE(y, x, 2, 2, u, umax)
            e.count_auto()
            var v = e.te_bits()
            _ = v
        var a1 = perf_counter_ns()
        if r == 0 or Int(a1 - a0) < t_rebuild:
            t_rebuild = Int(a1 - a0)

    var t_sc = 0
    for r in range(REPS):
        var a0 = perf_counter_ns()
        for u in range(1, umax + 1):
            var v = te_naive_packed(y, x, 2, 2, u)
            _ = v
        var a1 = perf_counter_ns()
        if r == 0 or Int(a1 - a0) < t_sc:
            t_sc = Int(a1 - a0)

    print("     set_lag reuse  :", padl(ms(t_scan), 11), "ms")
    print("     rebuild per lag:", padl(ms(t_rebuild), 11), "ms")
    print("     scalar/packed  :", padl(ms(t_sc), 11), "ms")
    print("     reuse vs scalar:",
          f6(Float64(t_sc) / Float64(t_scan)) + "x",
          "  peak u =", peak_u, " TE =", f6(peak_v))
    print("     (l = 2 means the window at u covers x[n-u+1] and x[n-u],")
    print("      so a driver at lag 2 shows at both u = 1 and u = 2)")

    # ------------------------------------------------------------------ F ---
    # The workload the packed representation is really for: screen many
    # candidate sources against one target.  k = l = 1 is the usual setting
    # for binary data, which is exactly where the AND-tree is strongest, and
    # all P sources together stay inside L2 instead of spilling to RAM.
    print("")
    print("[F] network screen: 64 candidate sources -> 1 target, k=l=u=1")
    var nsrc = 64
    var nn = 1000000
    var srcs = List[BitSeq]()
    for _i in range(nsrc):
        srcs.append(iid_packed(nn, rng))
    # Only source 7 actually drives the target; the rest are decoys.
    var driver = 7
    var yt = List[Int](length=nn, fill=0)
    yt[0] = rng.bit(0.5)
    for t in range(1, nn):
        yt[t] = srcs[driver].get(t - 1) ^ rng.bit(0.2)
    var ytb = BitSeq(yt)
    var ytl = yt.copy()
    var srcs_l = List[List[Int]]()
    for i in range(nsrc):
        var tmp = List[Int](length=nn, fill=0)
        for t in range(nn):
            tmp[t] = srcs[i].get(t)
        srcs_l.append(tmp^)

    print("     packed:", nsrc * nn // 8 // 1024, "KB total for all sources;",
          "List[Int]:", nsrc * nn * 8 // 1024 // 1024, "MB")
    var tb = 0
    var tp = 0
    var found = -1
    var found_v = -1.0
    for r in range(REPS):
        var a0 = perf_counter_ns()
        for i in range(nsrc):
            var est = BitTE(ytb, srcs[i], 1, 1, 1)
            est.count_auto()
            var v = est.te_bits()
            if v > found_v:
                found_v = v
                found = i
        var a1 = perf_counter_ns()
        if r == 0 or Int(a1 - a0) < tb:
            tb = Int(a1 - a0)

        var b0 = perf_counter_ns()
        for i in range(nsrc):
            var v = te_naive(ytl, srcs_l[i], 1, 1, 1)
            _ = v
        var b1 = perf_counter_ns()
        if r == 0 or Int(b1 - b0) < tp:
            tp = Int(b1 - b0)

    print("     AND-tree over packed :", padl(ms(tb), 11), "ms")
    print("     scalar over List[Int]:", padl(ms(tp), 11), "ms")
    print("     speedup:", f6(Float64(tp) / Float64(tb)) + "x",
          "  strongest source =", found, "(planted", String(driver) + "),",
          "TE =", f6(found_v), " expected", f6(1.0 - 0.721928))
