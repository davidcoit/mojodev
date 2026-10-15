# gpurun.mojo -- drive the GPU all-pairs kernel, check it, and time it.
#
#   /workspaces/mojodev/scratch/.venv-max/bin/mojo build -O3 gpurun.mojo -o gpurun
#   ./gpurun

from std.math import log2
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from bitte import BitSeq, BitTE, f6
from gen import Rng, iid, iid_packed, channel
from gpunet import net_te_kernel, TPB, NSTATES
from network import NetTE


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


def te_from_counts(c: List[Int], off: Int, m: Int) -> Float64:
    """TE in bits from the eight (y', y, x) counts, k = l = 1."""
    var n3 = List[Int](length=4, fill=0)    # (y, x)
    var n1 = List[Int](length=4, fill=0)    # (y', y)
    var n2 = List[Int](length=2, fill=0)    # (y)
    for s in range(8):
        var v = c[off + s]
        if v == 0:
            continue
        var f = s >> 2
        var hg = s & 3
        var h = hg >> 1
        n3[hg] += v
        n1[(f << 1) | h] += v
        n2[h] += v
    var total = 0.0
    for s in range(8):
        var v = c[off + s]
        if v == 0:
            continue
        var f = s >> 2
        var hg = s & 3
        var h = hg >> 1
        var num = Float64(v) * Float64(n2[h])
        var den = Float64(n3[hg]) * Float64(n1[(f << 1) | h])
        total += Float64(v) * log2(num / den)
    return total / Float64(m)


def main() raises:
    comptime assert has_accelerator(), "needs a GPU"

    var rng = Rng(90210)
    var p = 64
    var n = 1000000
    var u = 1

    print("GPU all-pairs transfer entropy, k = l = 1, lag", u)
    print("P =", p, "signals of", n, "samples ->", p * (p - 1),
          "ordered pairs")

    # A planted chain plus independent decoys: 0->1->2 and 3->4.
    var sigs = List[BitSeq]()
    var s0 = iid(n, 0.5, rng)
    var s1 = channel(s0, u, 0.05, rng)
    var s2 = channel(s1, u, 0.30, rng)
    var s3 = iid(n, 0.5, rng)
    var s4 = channel(s3, u, 0.15, rng)
    sigs.append(BitSeq(s0))
    sigs.append(BitSeq(s1))
    sigs.append(BitSeq(s2))
    sigs.append(BitSeq(s3))
    sigs.append(BitSeq(s4))
    for _i in range(p - 5):
        sigs.append(iid_packed(n, rng))

    var wps = len(sigs[0].w)
    var n0 = 0 if 0 > u - 1 else u - 1
    var m = n - 1 - n0
    var nw = (m + 63) // 64
    print("packed:", p * wps * 8 // 1024, "KB total  (as List[Int]:",
          p * n * 8 // 1024 // 1024, "MB)")
    print("")

    # ---------------------------------------------------------------- GPU ---
    var ctx = DeviceContext()
    var total_words = p * wps
    var sig_layout = row_major(total_words)
    var out_layout = row_major(p * p * NSTATES)

    var sig_buf = ctx.enqueue_create_buffer[DType.uint64](total_words)
    var out_buf = ctx.enqueue_create_buffer[DType.int32](p * p * NSTATES)

    # Flatten the packed signals, then one upload.
    var flat = List[UInt64](length=total_words, fill=0)
    for i in range(p):
        for w in range(wps):
            flat[i * wps + w] = sigs[i].w[w]

    # Copy straight out of the List. A pinned host buffer
    # (enqueue_create_host_buffer) costs ~0.3 s/MB under WSL2 here, which for
    # 7.8 MB is 2.3 s -- 500x the kernel itself.
    var t_up0 = perf_counter_ns()
    ctx.enqueue_copy(sig_buf, flat.unsafe_ptr())
    ctx.synchronize()
    var t_up1 = perf_counter_ns()

    var sig_t = TileTensor(sig_buf, sig_layout)
    var out_t = TileTensor(out_buf, out_layout)
    comptime kernel = net_te_kernel[type_of(sig_layout), type_of(out_layout)]

    # Warm up, then time.
    out_buf.enqueue_fill(0)
    ctx.enqueue_function[kernel](
        sig_t, out_t, Int32(p), Int32(nw), Int32(m), Int32(n0), Int32(u),
        Int32(wps), grid_dim=p * p, block_dim=TPB,
    )
    ctx.synchronize()

    var best = 0
    for r in range(5):
        out_buf.enqueue_fill(0)
        ctx.synchronize()
        var k0 = perf_counter_ns()
        ctx.enqueue_function[kernel](
            sig_t, out_t, Int32(p), Int32(nw), Int32(m), Int32(n0), Int32(u),
            Int32(wps), grid_dim=p * p, block_dim=TPB,
        )
        ctx.synchronize()
        var k1 = perf_counter_ns()
        if r == 0 or Int(k1 - k0) < best:
            best = Int(k1 - k0)

    var gcounts = List[Int](length=p * p * NSTATES, fill=0)
    with out_buf.map_to_host() as h:
        var ht = TileTensor(h, out_layout)
        for i in range(p * p * NSTATES):
            gcounts[i] = Int(ht[i])

    print("upload", p * wps * 8 // 1024, "KB:", ms(t_up1 - t_up0), "ms")
    print("kernel (best of 5):", ms(best), "ms  for", p * (p - 1), "pairs")
    print("   per pair:", f6(Float64(best) / Float64(p * (p - 1)) / 1000.0),
          "us")

    # ---------------------------------------------------------------- CPU ---
    var c0 = perf_counter_ns()
    var net = NetTE(sigs, 1, 1, u, 64)
    net.run()
    var c1 = perf_counter_ns()
    print("")
    print("CPU NetTE (shared target masks):", ms(c1 - c0), "ms")

    var d0 = perf_counter_ns()
    var naive_te = List[Float64](length=p * p, fill=0.0)
    for src in range(p):
        for tgt in range(p):
            if src == tgt:
                continue
            var e = BitTE(sigs[tgt], sigs[src], 1, 1, u)
            e.count()
            naive_te[src * p + tgt] = e.te_bits()
    var d1 = perf_counter_ns()
    print("CPU per-pair BitTE             :", ms(d1 - d0), "ms")
    print("")
    print("speedups:  GPU vs CPU-NetTE",
          f6(Float64(c1 - c0) / Float64(best)) + "x",
          "   GPU vs CPU-per-pair",
          f6(Float64(d1 - d0) / Float64(best)) + "x")
    print("           CPU-NetTE vs CPU-per-pair",
          f6(Float64(d1 - d0) / Float64(c1 - c0)) + "x")

    # ------------------------------------------------------------- verify ---
    print("")
    print("verifying GPU counts against the CPU estimator")
    var worst = 0.0
    var bad_counts = 0
    for src in range(p):
        for tgt in range(p):
            if src == tgt:
                continue
            var off = (src * p + tgt) * NSTATES
            var tot = 0
            for s in range(NSTATES):
                tot += gcounts[off + s]
            if tot != m:
                bad_counts += 1
            var got = te_from_counts(gcounts, off, m)
            var want = net.te[src * p + tgt]
            var d = got - want
            var ad = -d if d < 0.0 else d
            if ad > worst:
                worst = ad
    print("   count tables summing to m:",
          p * (p - 1) - bad_counts, "/", p * (p - 1))
    print("   worst |GPU TE - CPU TE| =", worst)
    if bad_counts == 0 and worst < 1e-12:
        print("  PASS GPU matches CPU exactly")
    else:
        print("  FAIL GPU disagrees with CPU")

    print("")
    print("planted edges recovered on GPU:")
    print("   0->1 (eps .05):", f6(te_from_counts(gcounts, (0 * p + 1) * NSTATES, m)),
          " expected", f6(1.0 - 0.286397))
    print("   1->2 (eps .30):", f6(te_from_counts(gcounts, (1 * p + 2) * NSTATES, m)))
    print("   3->4 (eps .15):", f6(te_from_counts(gcounts, (3 * p + 4) * NSTATES, m)))
    var strongest_absent = -1.0
    var sa_s = 0
    var sa_t = 0
    for src in range(p):
        for tgt in range(p):
            if src == tgt:
                continue
            var planted = (
                (src == 0 and tgt == 1)
                or (src == 1 and tgt == 2)
                or (src == 3 and tgt == 4)
            )
            if planted:
                continue
            var v = te_from_counts(gcounts, (src * p + tgt) * NSTATES, m)
            if v > strongest_absent:
                strongest_absent = v
                sa_s = src
                sa_t = tgt
    print("   strongest absent edge:", f6(strongest_absent),
          " at", sa_s, "->", sa_t)

    # ------------------------------------------------------ scaling in P ---
    # The pair count grows as P^2, so this is where GPU parallelism earns its
    # keep: every pair is an independent block with no cross-block traffic.
    print("")
    print("[B] GPU scaling in network size, n =", n, ", k = l = 1")
    print(pad("      P", 8), padl("pairs", 9), padl("kernel ms", 12),
          padl("us/pair", 10), padl("dev MB", 9), padl("CPU est s", 11))
    var ps = [8, 16, 32, 64, 128, 256]
    var us_at_64 = Float64(best) / Float64(p * (p - 1)) / 1000.0
    for pi in range(len(ps)):
        var pp2 = ps[pi]
        if pp2 > p:
            # Reuse the uploaded signals cyclically so the device buffer and
            # the upload are not re-measured; only the pair count changes.
            pass
        var tw2 = pp2 * wps
        var sl2 = row_major(tw2)
        var ol2 = row_major(pp2 * pp2 * NSTATES)
        var sb2 = ctx.enqueue_create_buffer[DType.uint64](tw2)
        var ob2 = ctx.enqueue_create_buffer[DType.int32](pp2 * pp2 * NSTATES)
        var f2 = List[UInt64](length=tw2, fill=0)
        for i in range(pp2):
            var srcsig = i % p
            for w in range(wps):
                f2[i * wps + w] = sigs[srcsig].w[w]
        ctx.enqueue_copy(sb2, f2.unsafe_ptr())
        ob2.enqueue_fill(0)
        ctx.synchronize()
        var st2 = TileTensor(sb2, sl2)
        var ot2 = TileTensor(ob2, ol2)
        comptime k2 = net_te_kernel[type_of(sl2), type_of(ol2)]
        ctx.enqueue_function[k2](
            st2, ot2, Int32(pp2), Int32(nw), Int32(m), Int32(n0), Int32(u),
            Int32(wps), grid_dim=pp2 * pp2, block_dim=TPB,
        )
        ctx.synchronize()
        var b2 = 0
        for r in range(3):
            var q0 = perf_counter_ns()
            ctx.enqueue_function[k2](
                st2, ot2, Int32(pp2), Int32(nw), Int32(m), Int32(n0),
                Int32(u), Int32(wps), grid_dim=pp2 * pp2, block_dim=TPB,
            )
            ctx.synchronize()
            var q1 = perf_counter_ns()
            if r == 0 or Int(q1 - q0) < b2:
                b2 = Int(q1 - q0)
        var npairs = pp2 * (pp2 - 1)
        var cpu_s = us_at_64 * 639.6 * Float64(npairs) / 1000000.0
        print(pad(String(pp2), 8), padl(String(npairs), 9),
              padl(ms(b2), 12),
              padl(f6(Float64(b2) / Float64(npairs) / 1000.0), 10),
              padl(String(tw2 * 8 // 1024 // 1024), 9),
              padl(f6(cpu_s), 11))
    print("    (CPU est s = per-pair CPU rate measured above, extrapolated)")
