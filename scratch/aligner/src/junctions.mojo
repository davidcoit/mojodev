"""Junction table for the second alignment pass.

Pass 1 aligns every read de novo.  Junctions seen in confident alignments (unique
MAPQ, long flanks on both sides) and supported by several reads are collected
here, sorted, and handed back to the kernel so that reads with very short
overhangs can be placed across *known* junctions (STAR-style 2-pass).
"""
from std.collections import Dict

from kernels import OUT_STRIDE, OUT_OPS, O_STATUS, O_POS, O_NOPS, O_MAPQ, OP_N, OP_M


struct JunctionDB(Movable):
    var n: Int
    var donor: List[UInt32]  # intron start (global), ascending
    var acc_of_donor: List[UInt32]  # matching intron end (exclusive)
    var acc: List[UInt32]  # intron end, ascending
    var donor_of_acc: List[UInt32]

    def __init__(out self, n: Int):
        # arrays always have >= 1 slot so the device buffers are never empty
        var m = max(n, 1)
        self.n = n
        self.donor = List[UInt32](length=m, fill=0)
        self.acc_of_donor = List[UInt32](length=m, fill=0)
        self.acc = List[UInt32](length=m, fill=0)
        self.donor_of_acc = List[UInt32](length=m, fill=0)


def radix_sort(mut keys: List[UInt64]):
    """LSD radix sort, 4 x 16-bit digits."""
    var n = len(keys)
    var tmp = List[UInt64](length=n, fill=0)
    var cnt = List[Int](length=65537, fill=0)
    for pass_ in range(4):
        var shift = UInt64(pass_ * 16)
        for i in range(65537):
            cnt[i] = 0
        for i in range(n):
            cnt[Int((keys[i] >> shift) & 0xFFFF) + 1] += 1
        for i in range(65536):
            cnt[i + 1] += cnt[i]
        for i in range(n):
            var dgt = Int((keys[i] >> shift) & 0xFFFF)
            tmp[cnt[dgt]] = keys[i]
            cnt[dgt] += 1
        for i in range(n):
            keys[i] = tmp[i]


def collect_junctions(
    records: List[Int32], n_reads: Int, min_support: Int, min_flank: Int
) raises -> JunctionDB:
    var counts = Dict[UInt64, Int32]()
    for r in range(n_reads):
        var rec = records.unsafe_ptr() + r * OUT_STRIDE
        if Int(rec[O_STATUS]) != 1 or Int(rec[O_MAPQ]) < 10:
            continue
        var n_ops = Int(rec[O_NOPS])
        var ref_pos = Int(rec[O_POS])
        for t in range(n_ops):
            var v = Int(rec[OUT_OPS + t])
            var op = v & 7
            var length = v >> 3
            if op == OP_N:
                if t > 0 and t + 1 < n_ops:
                    var prev = Int(rec[OUT_OPS + t - 1])
                    var nxt = Int(rec[OUT_OPS + t + 1])
                    if (prev & 7) == OP_M and (nxt & 7) == OP_M and (prev >> 3) >= min_flank and (nxt >> 3) >= min_flank:
                        var key = (UInt64(ref_pos) << 32) | UInt64(ref_pos + length)
                        if key in counts:
                            counts[key] = counts[key] + 1
                        else:
                            counts[key] = 1
                ref_pos += length
            elif op == OP_M or op == 2:  # M or D advance the reference
                ref_pos += length

    var by_donor = List[UInt64]()
    var by_acc = List[UInt64]()
    for e in counts.items():
        if Int(e.value) >= min_support:
            var s = e.key >> 32
            var a = e.key & 0xFFFFFFFF
            by_donor.append(e.key)
            by_acc.append((a << 32) | s)
    radix_sort(by_donor)
    radix_sort(by_acc)
    var db = JunctionDB(len(by_donor))
    for i in range(len(by_donor)):
        db.donor[i] = UInt32(by_donor[i] >> 32)
        db.acc_of_donor[i] = UInt32(by_donor[i] & 0xFFFFFFFF)
        db.acc[i] = UInt32(by_acc[i] >> 32)
        db.donor_of_acc[i] = UInt32(by_acc[i] & 0xFFFFFFFF)
    return db^
