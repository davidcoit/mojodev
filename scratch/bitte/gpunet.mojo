# gpunet.mojo -- all-pairs transfer entropy on the GPU, k = l = 1, any lag.
#
# Build and run with the max venv's mojo (the stdlib alone has no GPU API):
#   /workspaces/mojodev/scratch/.venv-max/bin/mojo build -O3 gpunet.mojo -o gpunet
#
# MAPPING
# -------
# One block per ordered pair (source -> target); P*P blocks, the P diagonal
# ones exit immediately. Each thread strides over the time axis a word at a
# time, keeping the whole 3-plane state tree in registers and accumulating the
# eight counts of (y', y, x) in eight more registers. At the end the block
# folds its threads together through eight shared-memory bins and writes eight
# Int32 to global.
#
# Consequences of that choice:
#  - No global atomics anywhere. Each pair owns its eight output slots, so the
#    only contention is 8 shared-memory adds per thread.
#  - No barriers in the hot loop, and no cross-thread communication at all
#    until the final fold -- every thread's words are independent, and the
#    complement identity (mask & ~P == mask ^ (mask & P)) plus the count
#    subtraction both hold per word, so a thread needs no neighbour's totals.
#  - It deliberately gives up the CPU version's trick of sharing the 2^(k+1)
#    target masks across all sources. That would halve the arithmetic (4 plane
#    passes per pair instead of 8) but forces either a source loop inside the
#    block -- which needs P*8 shared bins and P*8 atomics per thread -- or
#    global atomics. On a GPU the arithmetic is free and the atomics are not,
#    so recomputing the target masks per pair is the better trade.
#
# Bit-packing is what makes this work at all: P signals of n samples are
# P*n/8 bytes, so a 64-signal, 1M-sample network is 7.8 MB and lives in L2
# while 4032 blocks hammer it. The same data as List[Int] would be 488 MB.

from std.atomic import Atomic
from std.bit import pop_count
from std.memory import stack_allocation
from std.sys import has_accelerator
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major

comptime TPB = 256          # threads per block
comptime NSTATES = 8        # 2^(1+k+l) for k = l = 1


def plane_word[
    LT: TensorLayout
](
    sigs: TileTensor[DType.uint64, LT, MutAnyOrigin],
    sig_base: Int,
    bitpos: Int,
) -> UInt64:
    """One 64-bit window of a signal starting at an arbitrary bit offset.

    The two-word funnel shift, written so both shift counts stay below 64:
    `(hi << 1) << (63 - r)` is zero when r == 0 instead of undefined. Every
    BitSeq carries a guard word, so reading q+1 is always in range.
    """
    comptime assert sigs.flat_rank == 1, "signal buffer is 1-D"
    var q = bitpos >> 6
    var r = UInt64(bitpos & 63)
    var lo = rebind[Scalar[DType.uint64]](sigs[sig_base + q])
    var hi = rebind[Scalar[DType.uint64]](sigs[sig_base + q + 1])
    return (lo >> r) | ((hi << 1) << (63 - r))


def net_te_kernel[
    LS: TensorLayout, LO: TensorLayout
](
    sigs: TileTensor[DType.uint64, LS, MutAnyOrigin],
    out_counts: TileTensor[DType.int32, LO, MutAnyOrigin],
    p: Int32,
    nw: Int32,
    m: Int32,
    n0: Int32,
    u: Int32,
    wps: Int32,
):
    comptime assert sigs.flat_rank == 1, "signal buffer is 1-D"
    comptime assert out_counts.flat_rank == 1, "count buffer is 1-D"

    var pair = Int(block_idx.x)
    var pp = Int(p)
    var src = pair // pp
    var tgt = pair % pp

    var sh = stack_allocation[
        NSTATES, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    if tid < NSTATES:
        sh[unsafe_offset=tid] = 0
    barrier()

    if src != tgt:
        var sbase = src * Int(wps)
        var tbase = tgt * Int(wps)
        var f_off = Int(n0) + 1
        var h_off = Int(n0)
        var g_off = Int(n0) - Int(u) + 1

        # Eight private accumulators, index (y' << 2) | (y << 1) | x.
        var a0: Int32 = 0
        var a1: Int32 = 0
        var a2: Int32 = 0
        var a3: Int32 = 0
        var a4: Int32 = 0
        var a5: Int32 = 0
        var a6: Int32 = 0
        var a7: Int32 = 0

        var w = tid
        while w < Int(nw):
            var bitbase = w << 6
            var valid = Int(m) - bitbase
            var root = ~UInt64(0)
            if valid < 64:
                root = (UInt64(1) << UInt64(valid)) - 1
            else:
                valid = 64

            var fp = plane_word(sigs, tbase, f_off + bitbase)
            var hp = plane_word(sigs, tbase, h_off + bitbase)
            var gp = plane_word(sigs, sbase, g_off + bitbase)

            # Split on the future bit, then on the target history bit. The
            # four masks partition this word, so each complement is one XOR
            # and each complement's size is one subtraction.
            var mf1 = root & fp
            var mf0 = root ^ mf1
            var pf1 = Int32(pop_count(mf1))
            var pf0 = Int32(valid) - pf1

            var m11 = mf1 & hp
            var m10 = mf1 ^ m11
            var p11 = Int32(pop_count(m11))
            var p10 = pf1 - p11

            var m01 = mf0 & hp
            var m00 = mf0 ^ m01
            var p01 = Int32(pop_count(m01))
            var p00 = pf0 - p01

            # Deepest level needs counts only, never a mask: AND for x = 1,
            # subtract for x = 0.
            var c11 = Int32(pop_count(m11 & gp))
            var c10 = Int32(pop_count(m10 & gp))
            var c01 = Int32(pop_count(m01 & gp))
            var c00 = Int32(pop_count(m00 & gp))

            a7 += c11
            a6 += p11 - c11
            a5 += c10
            a4 += p10 - c10
            a3 += c01
            a2 += p01 - c01
            a1 += c00
            a0 += p00 - c00

            w += Int(block_dim.x)

        _ = Atomic.fetch_add(sh.unsafe_offset(0), a0)
        _ = Atomic.fetch_add(sh.unsafe_offset(1), a1)
        _ = Atomic.fetch_add(sh.unsafe_offset(2), a2)
        _ = Atomic.fetch_add(sh.unsafe_offset(3), a3)
        _ = Atomic.fetch_add(sh.unsafe_offset(4), a4)
        _ = Atomic.fetch_add(sh.unsafe_offset(5), a5)
        _ = Atomic.fetch_add(sh.unsafe_offset(6), a6)
        _ = Atomic.fetch_add(sh.unsafe_offset(7), a7)

    barrier()
    if tid < NSTATES:
        out_counts[pair * NSTATES + tid] = sh[unsafe_offset=tid]
