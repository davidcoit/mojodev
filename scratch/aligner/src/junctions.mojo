"""Junction table for the second alignment pass.

Pass 1 aligns every read de novo.  Junctions seen in confident alignments (unique
MAPQ, long flanks on both sides) and supported by several reads are collected
here, sorted, and handed back to the kernel so that reads with very short
overhangs can be placed across *known* junctions (STAR-style 2-pass).
"""
from std.collections import Dict

from nt4 import PU8, nt4_get
from kernels import REC_STRIDE, OUT_OPS, O_STATUS, O_POS, O_NOPS, O_MAPQ, OP_N, OP_M


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


struct JunctionCounter(Movable):
    """Accumulates splice junctions seen in confident alignments across chunks."""
    var counts: Dict[UInt64, Int32]

    def __init__(out self):
        self.counts = Dict[UInt64, Int32]()

    def add(mut self, records: List[Int32], n_reads: Int, min_flank: Int) raises:
        """Count junctions of candidate 0 of the first n_reads records.

        A junction counts when the alignment is unique (MAPQ >= 10) and the M blocks
        on both sides are at least min_flank long.
        """
        for r in range(n_reads):
            var rec = records.unsafe_ptr() + r * REC_STRIDE
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
                            if key in self.counts:
                                self.counts[key] = self.counts[key] + 1
                            else:
                                self.counts[key] = 1
                    ref_pos += length
                elif op == OP_M or op == 2:  # M or D advance the reference
                    ref_pos += length

    def build(self, min_support: Int, genome: PU8, n_bases: Int) -> JunctionDB:
        """Sorted junction table.

        A junction is kept if at least min_support reads show it AND either it has a
        canonical motif (GT..AG, CT..AC) or at least 3 reads show it, so single-read
        non-canonical junctions (mostly alignment noise) are not propagated.
        """
        var by_donor = List[UInt64]()
        var by_acc = List[UInt64]()
        for e in self.counts.items():
            var s = e.key >> 32
            var a = e.key & 0xFFFFFFFF
            var keep = Int(e.value) >= min_support
            if keep and Int(e.value) < 3:
                var si = Int(s)
                var ai = Int(a)
                var canonical = False
                if si >= 0 and ai + 1 < n_bases and ai - si >= 4:
                    var d1 = nt4_get(genome, si)
                    var d2 = nt4_get(genome, si + 1)
                    var a1 = nt4_get(genome, ai - 2)
                    var a2 = nt4_get(genome, ai - 1)
                    canonical = (d1 == 2 and d2 == 3 and a1 == 0 and a2 == 2) or (d1 == 1 and d2 == 3 and a1 == 0 and a2 == 1)
                keep = canonical
            if keep:
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
