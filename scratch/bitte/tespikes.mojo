# tespikes.mojo -- transfer entropy over a real spike-train network.
#
#   <venv>/bin/python -I prep_spikes.py data/fluorescence_*.txt data/spikes.bits
#   <venv>/bin/mojo build -O3 tespikes.mojo -o tespikes && ./tespikes data/spikes.bits
#
# Reads the packed bits written by prep_spikes.py, then estimates TE for every
# ordered pair at a short, a medium and a long lag, on both CPU and GPU.
#
# THE NOISE FLOOR
# ---------------
# A TE number on real data means nothing without knowing what zero looks like,
# and the plug-in estimator is positively biased, so "> 0" is not a test. The
# null here is a cyclic rotation of the source: it destroys the source's timing
# relative to the target while preserving its marginal rate and its entire
# autocorrelation, so the spread of those values is this data's bias floor.
#
# It comes for free with no change to the estimator. The signal list handed to
# NetTE is [originals..., rotated...], so one all-pairs run yields both the
# real edge i->j and its own surrogate (rotated i)->j. It costs 4x the pairs,
# which against a 100x-plus speedup is nothing.

from std.math import log2, sqrt
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from bitte import BitSeq, slice_into, f6
from gpunet import net_te_kernel, NSTATES, TPB
from network import NetTE

comptime MAGIC = 20261015
comptime FRAME_HZ = 50.0        # Connectomics simulated frames are 50 Hz


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


def load_bits(path: String) raises -> List[BitSeq]:
    """Read the packed spike file straight into BitSeq buffers."""
    var fh = open(path, "r")
    var raw = fh.read_bytes()
    fh.close()
    if len(raw) < 32:
        raise Error("file too short to hold a header")

    var hdr = List[Int](length=4, fill=0)
    for h in range(4):
        var v: UInt64 = 0
        for b in range(8):
            v |= UInt64(Int(raw[h * 8 + b])) << UInt64(8 * b)
        hdr[h] = Int(v)
    if hdr[0] != MAGIC:
        raise Error("bad magic; not a prep_spikes.py file")
    var p = hdr[1]
    var n = hdr[2]
    var wps = hdr[3]
    if 32 + p * wps * 8 != len(raw):
        raise Error("header does not match payload length")

    var sigs = List[BitSeq]()
    for j in range(p):
        var s = BitSeq(n)
        if len(s.w) != wps:
            raise Error("wps in file disagrees with BitSeq layout")
        var base = 32 + j * wps * 8
        for w in range(wps):
            var v: UInt64 = 0
            var o = base + w * 8
            for b in range(8):
                v |= UInt64(Int(raw[o + b])) << UInt64(8 * b)
            s.w[w] = v
        sigs.append(s^)
    print("loaded", p, "signals x", n, "samples from", path)
    return sigs^


def rotated(sig: BitSeq, r: Int) raises -> BitSeq:
    """Cyclic rotation by r samples: bit i becomes sig[(i + r) % n]."""
    var d = sig.doubled()
    var out = BitSeq(sig.n)
    slice_into(d, r % sig.n, sig.n, out.w, 0)
    return out^


def density(sig: BitSeq) -> Float64:
    var c = 0
    for i in range(sig.n):
        c += sig.get(i)
    return Float64(c) / Float64(sig.n)


def main() raises:
    var args = argv()
    var path = String("data/spikes_z25.bits")
    if len(args) > 1:
        path = String(args[1])

    var real = load_bits(path)
    var p = len(real)
    var n = real[0].n
    print("  density of signal 0:", f6(density(real[0])),
          " -> ", f6(density(real[0]) * FRAME_HZ), "Hz at", FRAME_HZ, "Hz")

    # [originals..., rotated by n/2...] so one pass gives edges and their null.
    var both = List[BitSeq]()
    for i in range(p):
        both.append(real[i].copy())
    for i in range(p):
        both.append(rotated(real[i], n // 2))
    var p2 = len(both)
    print("  signal list for the run:", p2,
          "(", p, "real +", p, "rotation surrogates )")
    print("")

    # Short / medium / long, in frames and milliseconds.
    var lags = [1, 5, 25]
    var names = [String("short"), String("medium"), String("long")]

    var ctx = DeviceContext()
    var wps = len(both[0].w)
    var total_words = p2 * wps
    var sig_layout = row_major(total_words)
    var out_layout = row_major(p2 * p2 * NSTATES)
    var sig_buf = ctx.enqueue_create_buffer[DType.uint64](total_words)
    var out_buf = ctx.enqueue_create_buffer[DType.int32](p2 * p2 * NSTATES)

    var flat = List[UInt64](length=total_words, fill=0)
    for i in range(p2):
        for w in range(wps):
            flat[i * wps + w] = both[i].w[w]
    var up0 = perf_counter_ns()
    ctx.enqueue_copy(sig_buf, flat.unsafe_ptr())
    ctx.synchronize()
    var up1 = perf_counter_ns()
    var sig_t = TileTensor(sig_buf, sig_layout)
    var out_t = TileTensor(out_buf, out_layout)
    comptime kernel = net_te_kernel[type_of(sig_layout), type_of(out_layout)]

    print("packed upload:", f6(Float64(total_words * 8) / 1048576.0), "MB in",
          ms(up1 - up0), "ms")
    print("")
    print("=== timing, ", p2 * (p2 - 1), " ordered pairs per lag ===")
    print(pad("  lag", 18), padl("CPU NetTE ms", 14), padl("GPU ms", 10),
          padl("speedup", 10), padl("GPU us/pair", 13))

    var gpu_counts = List[Int](length=p2 * p2 * NSTATES, fill=0)
    var summary_lag = List[Int]()
    var summary_edges = List[Int]()
    var summary_floor = List[Float64]()
    var summary_max = List[Float64]()

    for li in range(len(lags)):
        var u = lags[li]

        # ---- CPU ----
        var c0 = perf_counter_ns()
        var net = NetTE(both, 1, 1, u, 64)
        net.run()
        var c1 = perf_counter_ns()

        # ---- GPU ----
        var n0 = 0 if 0 > u - 1 else u - 1
        var m = n - 1 - n0
        var nw = (m + 63) // 64
        out_buf.enqueue_fill(0)
        ctx.enqueue_function[kernel](
            sig_t, out_t, Int32(p2), Int32(nw), Int32(m), Int32(n0),
            Int32(u), Int32(wps), grid_dim=p2 * p2, block_dim=TPB,
        )
        ctx.synchronize()
        var best = 0
        for r in range(3):
            out_buf.enqueue_fill(0)
            ctx.synchronize()
            var g0 = perf_counter_ns()
            ctx.enqueue_function[kernel](
                sig_t, out_t, Int32(p2), Int32(nw), Int32(m), Int32(n0),
                Int32(u), Int32(wps), grid_dim=p2 * p2, block_dim=TPB,
            )
            ctx.synchronize()
            var g1 = perf_counter_ns()
            if r == 0 or Int(g1 - g0) < best:
                best = Int(g1 - g0)
        with out_buf.map_to_host() as h:
            var ht = TileTensor(h, out_layout)
            for i in range(p2 * p2 * NSTATES):
                gpu_counts[i] = Int(ht[i])

        var npairs = p2 * (p2 - 1)
        var label = names[li] + " u=" + String(u) + " ("
        label = label + String(f6(Float64(u) * 1000.0 / FRAME_HZ)[byte=0:5]) + " ms)"
        print(pad("  " + label, 18),
              padl(ms(c1 - c0), 14), padl(ms(best), 10),
              padl(f6(Float64(c1 - c0) / Float64(best)) + "x", 10),
              padl(f6(Float64(best) / Float64(npairs) / 1000.0), 13))

        # ---- agreement, then the noise floor ----
        var worst = 0.0
        for src in range(p2):
            for tgt in range(p2):
                if src == tgt:
                    continue
                var got = te_from_counts(
                    gpu_counts, (src * p2 + tgt) * NSTATES, m
                )
                var want = net.te[src * p2 + tgt]
                var d = got - want
                var ad = -d if d < 0.0 else d
                if ad > worst:
                    worst = ad
        if worst > 1e-12:
            print("      WARNING cpu/gpu disagree by", worst)

        # Surrogate values: rotated source (index p + i) onto real target j.
        var smax = 0.0
        var ssum = 0.0
        var ssq = 0.0
        var scnt = 0
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                var v = net.te[(p + i) * p2 + j]
                ssum += v
                ssq += v * v
                scnt += 1
                if v > smax:
                    smax = v
        var smean = ssum / Float64(scnt)
        var svar = ssq / Float64(scnt) - smean * smean
        var ssd = sqrt(svar) if svar > 0.0 else 0.0
        var floor = smax

        var above = 0
        var rmax = 0.0
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                var v = net.te[i * p2 + j]
                if v > floor:
                    above += 1
                if v > rmax:
                    rmax = v
        summary_lag.append(u)
        summary_edges.append(above)
        summary_floor.append(floor)
        summary_max.append(rmax)

        print("      surrogate null: mean", f6(smean), " sd", f6(ssd),
              " max", f6(smax))
        print("      real edges: max", f6(rmax), " above the null max:",
              above, "/", p * (p - 1))

    # ------------------------------------------------------------------
    print("")
    print("=== lag profile ===")
    print(pad("  lag (frames)", 16), padl("ms", 8), padl("edges", 8),
          padl("null max", 11), padl("strongest", 11))
    for i in range(len(summary_lag)):
        print(pad("  " + String(summary_lag[i]), 16),
              padl(String(f6(Float64(summary_lag[i]) * 1000.0 / FRAME_HZ)[byte=0:5]), 8),
              padl(String(summary_edges[i]), 8),
              padl(f6(summary_floor[i]), 11),
              padl(f6(summary_max[i]), 11))

    # ------------------------------------------------------------------
    # Strongest edges at the short lag, recomputed so the table is explicit.
    print("")
    print("=== strongest edges at the short lag (u = 1) ===")
    var net1 = NetTE(both, 1, 1, 1, 64)
    net1.run()
    var floor1 = 0.0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            var v = net1.te[(p + i) * p2 + j]
            if v > floor1:
                floor1 = v
    print("  (null max =", f6(floor1), "bits)")
    print(pad("  source -> target", 20), padl("TE bits", 11),
          padl("x null max", 12))
    for rank in range(12):
        var bi = -1
        var bj = -1
        var bv = -1.0
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                var v = net1.te[i * p2 + j]
                if v > bv:
                    bv = v
                    bi = i
                    bj = j
        if bi < 0:
            break
        print(pad("  " + String(bi) + " -> " + String(bj), 20),
              padl(f6(bv), 11),
              padl(f6(bv / floor1) + "x", 12))
        net1.te[bi * p2 + bj] = -1.0   # strike it out and take the next


def te_from_counts(c: List[Int], off: Int, m: Int) -> Float64:
    """TE in bits from the eight (y', y, x) counts, k = l = 1."""
    var n3 = List[Int](length=4, fill=0)
    var n1 = List[Int](length=4, fill=0)
    var n2 = List[Int](length=2, fill=0)
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
