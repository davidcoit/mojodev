"""Canonical (k, w) minimizers over a 4-bit packed sequence.

The scanner is shared by the host (reference indexing) and the GPU read kernel,
so a read and the genome select identical minimizers wherever they agree.

Each output minimizer is (hash32, pos_and_strand):
    pos_and_strand = start_of_kmer | (strand << 31)
strand is 1 when the reverse complement was the smaller (canonical) k-mer.
K is odd, so a k-mer is never its own reverse complement.
"""
from nt4 import PU8, PU32, nt4_get

comptime MM_K = 15
comptime MM_W = 8
comptime MM_MASK: UInt64 = (UInt64(1) << UInt64(2 * MM_K)) - 1
comptime MM_NONE: UInt64 = 0xFFFFFFFFFFFFFFFF


@always_inline
def mm_hash(x: UInt64) -> UInt32:
    """Invertible-style 64->32 bit mixer (splitmix64 finaliser)."""
    var z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return UInt32(z >> 32)


def scan_minimizers(
    seq: PU8,
    start: Int,
    end: Int,
    out_hash: PU32,
    out_pos: PU32,
    max_out: Int,
) -> Int:
    """Emit the minimizers of seq[start:end]; returns how many were written."""
    # Sliding window of the last MM_W entries held in registers (MM_W == 8).
    var r0 = MM_NONE
    var r1 = MM_NONE
    var r2 = MM_NONE
    var r3 = MM_NONE
    var r4 = MM_NONE
    var r5 = MM_NONE
    var r6 = MM_NONE
    var r7 = MM_NONE
    var fwd = UInt64(0)
    var rc = UInt64(0)
    var valid = 0
    var n_out = 0
    var last = MM_NONE
    var rc_shift = UInt64(2 * (MM_K - 1))
    for i in range(start, end):
        var c = nt4_get(seq, i)
        var entry = MM_NONE
        if c > 3:
            valid = 0
        else:
            fwd = ((fwd << 2) | UInt64(c)) & MM_MASK
            rc = (rc >> 2) | (UInt64(3 - c) << rc_shift)
            valid += 1
            if valid >= MM_K:
                var canon = fwd
                var strand = UInt64(0)
                if rc < fwd:
                    canon = rc
                    strand = 1
                var kstart = UInt64(i - MM_K + 1)
                entry = (UInt64(mm_hash(canon)) << 32) | (strand << 31) | kstart
        r0 = r1
        r1 = r2
        r2 = r3
        r3 = r4
        r4 = r5
        r5 = r6
        r6 = r7
        r7 = entry
        if (i - start) >= MM_K + MM_W - 2:
            var best = min(min(min(r0, r1), min(r2, r3)), min(min(r4, r5), min(r6, r7)))
            if best != MM_NONE and best != last:
                if n_out >= max_out:
                    return n_out
                out_hash[n_out] = UInt32(best >> 32)
                out_pos[n_out] = UInt32(best & 0xFFFFFFFF)
                n_out += 1
                last = best
    return n_out
