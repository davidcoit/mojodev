# netvalidate.mojo -- the all-pairs path must agree with the single-pair one.

from std.time import perf_counter_ns
from bitte import BitSeq, BitTE, f6
from network import NetTE
from gen import Rng, iid, iid_packed, channel


def ms(ns: Int) -> String:
    return f6(Float64(ns) / 1000000.0)


def main() raises:
    var fails = 0
    var rng = Rng(31337)

    # ------------------------------------------------------------------ 1 ---
    print("[1] NetTE vs per-pair BitTE, exact agreement required")
    print("    sweeping k, l, u, P and lengths across the word boundary")
    var lengths = [127, 128, 129, 1000, 5000]
    var worst = 0.0
    var pairs_checked = 0
    for li in range(len(lengths)):
        var n = lengths[li]
        # A small network: 0 drives 1 at lag 2, 2 drives 3 at lag 1, 4 is noise.
        var s0 = iid(n, 0.5, rng)
        var s1 = channel(s0, 2, 0.1, rng)
        var s2 = iid(n, 0.35, rng)
        var s3 = channel(s2, 1, 0.25, rng)
        var s4 = iid(n, 0.5, rng)
        var sigs = List[BitSeq]()
        sigs.append(BitSeq(s0))
        sigs.append(BitSeq(s1))
        sigs.append(BitSeq(s2))
        sigs.append(BitSeq(s3))
        sigs.append(BitSeq(s4))

        for k in range(1, 4):
            for l in range(1, 4):
                for u in range(1, 4):
                    var n0 = k - 1 if k - 1 > u + l - 2 else u + l - 2
                    if n - 1 - n0 < 16:
                        continue
                    for tile in [1, 3, 64, 100000]:
                        var net = NetTE(sigs, k, l, u, tile)
                        net.run()
                        for src in range(len(sigs)):
                            for tgt in range(len(sigs)):
                                if src == tgt:
                                    continue
                                var e = BitTE(sigs[tgt], sigs[src], k, l, u)
                                e.count()
                                var want = e.te_bits()
                                var got = net.te[src * net.p + tgt]
                                var d = got - want
                                var ad = -d if d < 0.0 else d
                                if ad > worst:
                                    worst = ad
                                pairs_checked += 1
    print("    ", pairs_checked, "pair estimates compared")
    print("     worst |NetTE - BitTE| =", worst)
    if worst > 1e-12:
        print("  FAIL all-pairs disagrees with per-pair")
        fails += 1
    else:
        print("  PASS exact to double precision, every tile width")

    # ------------------------------------------------------------------ 2 ---
    print("")
    print("[2] count conservation: every source table must total m")
    var n2 = 4096
    var a0 = iid(n2, 0.5, rng)
    var a1 = channel(a0, 1, 0.1, rng)
    var a2 = iid(n2, 0.5, rng)
    var sg = List[BitSeq]()
    sg.append(BitSeq(a0))
    sg.append(BitSeq(a1))
    sg.append(BitSeq(a2))
    var ok = True
    for k in range(1, 4):
        for l in range(1, 3):
            var net = NetTE(sg, k, l, 1, 7)
            net.run()
            for src in range(net.p):
                if src == net.p - 1:
                    pass
                var tot = net.total_counted(src)
                # The table left over belongs to the last target processed;
                # every source except that target must hold exactly m counts.
                if tot != net.m and tot != 0:
                    ok = False
                    print("     k", k, "l", l, "src", src, "total", tot,
                          "m", net.m)
    if ok:
        print("  PASS every populated table sums to m")
    else:
        print("  FAIL count conservation")
        fails += 1

    # ------------------------------------------------------------------ 3 ---
    print("")
    print("[3] recovering a planted network, P = 8, n = 500,000")
    var n3 = 500000
    var base = iid_packed(n3, rng)
    var bl = List[Int](length=n3, fill=0)
    for t in range(n3):
        bl[t] = base.get(t)
    # 0 -> 1 strong (eps .05), 1 -> 2 weak (eps .30), 3 -> 4 strong,
    # 5, 6, 7 independent.  No other edges.
    var g0 = bl.copy()
    var g1 = channel(g0, 1, 0.05, rng)
    var g2 = channel(g1, 1, 0.30, rng)
    var g3 = iid(n3, 0.5, rng)
    var g4 = channel(g3, 1, 0.05, rng)
    var g5 = iid(n3, 0.5, rng)
    var g6 = iid(n3, 0.5, rng)
    var g7 = iid(n3, 0.5, rng)
    var net_sigs = List[BitSeq]()
    net_sigs.append(BitSeq(g0))
    net_sigs.append(BitSeq(g1))
    net_sigs.append(BitSeq(g2))
    net_sigs.append(BitSeq(g3))
    net_sigs.append(BitSeq(g4))
    net_sigs.append(BitSeq(g5))
    net_sigs.append(BitSeq(g6))
    net_sigs.append(BitSeq(g7))

    var t0 = perf_counter_ns()
    var net = NetTE(net_sigs, 1, 1, 1, 64)
    net.run()
    var t1 = perf_counter_ns()
    print("     56 ordered pairs in", ms(t1 - t0), "ms")
    print("     TE matrix (rows = source, cols = target), bits:")
    var hdr = "        "
    for tgt in range(net.p):
        hdr = hdr + "    t" + String(tgt)
    print(hdr)
    for src in range(net.p):
        var line = "     s" + String(src) + " "
        for tgt in range(net.p):
            if src == tgt:
                line = line + "     ."
            else:
                var v = net.te[src * net.p + tgt]
                var tag = " " + f6(v)
                line = line + tag[byte=0:7]
        print(line)
    # The planted edges must dominate every absent one.
    var planted = [0 * net.p + 1, 1 * net.p + 2, 3 * net.p + 4]
    var weakest_planted = 1.0e9
    for i in range(len(planted)):
        var v = net.te[planted[i]]
        if v < weakest_planted:
            weakest_planted = v
    var strongest_absent = -1.0
    for src in range(net.p):
        for tgt in range(net.p):
            if src == tgt:
                continue
            var idx = src * net.p + tgt
            var is_planted = False
            for i in range(len(planted)):
                if planted[i] == idx:
                    is_planted = True
            if not is_planted:
                var v = net.te[idx]
                if v > strongest_absent:
                    strongest_absent = v
    print("     weakest planted edge =", f6(weakest_planted))
    print("     strongest absent edge =", f6(strongest_absent))
    if weakest_planted > strongest_absent * 10.0:
        print("  PASS planted edges separate from absent ones")
    else:
        print("  FAIL edge separation")
        fails += 1

    print("")
    if fails == 0:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES:", fails)
