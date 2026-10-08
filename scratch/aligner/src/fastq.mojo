"""Multi-threaded FASTQ reader that packs reads straight into 4-bit slots.

Three stages, each spread over the host's cores (none of this touches the GPU):

1. refill   - the next block of the file is read with several concurrent ranged reads
              (one serial read() runs at ~0.5 GB/s; concurrent ones scale).
2. index    - the block is cut into segments; each worker resynchronises to the first
              record start in its segment (a line starting '@' whose second-next line
              starts '+' with equal sequence/quality lengths) and records the offset of
              every record it owns.
3. parse    - templates are split across workers; each packs sequence bytes into 4-bit
              slots (two bases per byte via a lookup table) and copies the read name.

Records never straddle a refill: the unconsumed tail is moved to the front first.
The reader holds no open file handle (workers open the file for their own ranges).
"""
from max.algorithm import sync_parallelize

from nt4 import PU8, nt4_from_ascii
from kernels import MAXL, RSTRIDE

comptime BLOCK_BYTES = 128 * 1024 * 1024
comptime READ_PIECES = 16  # concurrent ranged reads per refill
comptime INDEX_SEGMENTS = 48
comptime PARSE_TASKS = 96
comptime NAME_STRIDE = 128  # bytes reserved per template name; longer names raise


struct FastqReader(Movable):
    var path: String
    var file_off: Int
    var buf: List[UInt8]
    var end: Int  # valid bytes in buf
    var block_end: Int  # end of the last indexed complete record
    var eof: Bool
    var starts: List[Int32]  # offsets of indexed records
    var n_rec: Int
    var cursor: Int  # next unconsumed record
    var seg_starts: List[Int32]
    var seg_counts: List[Int]
    var seg_ends: List[Int]
    var seg_cap: Int

    def __init__(out self, path: String) raises:
        self.path = path
        self.file_off = 0
        self.buf = List[UInt8](length=BLOCK_BYTES + 4096, fill=0)
        self.end = 0
        self.block_end = 0
        self.eof = False
        self.n_rec = 0
        self.cursor = 0
        self.seg_cap = BLOCK_BYTES // INDEX_SEGMENTS // 8 + 64
        self.seg_starts = List[Int32](length=INDEX_SEGMENTS * self.seg_cap, fill=0)
        self.seg_counts = List[Int](length=INDEX_SEGMENTS, fill=0)
        self.seg_ends = List[Int](length=INDEX_SEGMENTS, fill=0)
        self.starts = List[Int32](length=BLOCK_BYTES // 8 + 64, fill=0)
        self.load_block()

    def finished(self) -> Bool:
        return self.eof and self.cursor >= self.n_rec and self.block_end >= self.end

    def load_block(mut self) raises:
        """Keep the unconsumed tail, read the next block, and index its records."""
        var bp = self.buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var keep = self.end - self.block_end
        for i in range(keep):
            bp[i] = bp[self.block_end + i]
        var want = BLOCK_BYTES - keep
        var piece = (want + READ_PIECES - 1) // READ_PIECES
        var got = List[Int](length=READ_PIECES, fill=0)
        var gp = got.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var path = self.path
        var file_off = self.file_off
        var dst = bp + keep

        def read_piece(k: Int) raises {dst, gp, path, file_off, piece, want}:
            var lo = k * piece
            var n = min(piece, want - lo)
            if n <= 0:
                gp[k] = 0
                return
            var f = open(path, "r")
            _ = f.seek(file_off + lo, 0)
            gp[k] = Int(f.read(Span[UInt8, MutAnyOrigin](unsafe_ptr=dst + lo, length=n)))

        sync_parallelize(read_piece, READ_PIECES)
        # keep only the contiguous prefix (a short piece means end of file)
        var total = 0
        for k in range(READ_PIECES):
            var n = min(piece, want - k * piece)
            total += got[k]
            if got[k] < n:
                break
        if total < want:
            self.eof = True
        self.file_off += total
        self.end = keep + total
        self.index_block()

    def index_block(mut self) raises:
        var p = self.buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var end = self.end
        var seg = (end + INDEX_SEGMENTS - 1) // INDEX_SEGMENTS
        var cap = self.seg_cap
        var sp = self.seg_starts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var cp = self.seg_counts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var lp = self.seg_ends.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var at_eof = self.eof

        def scan(s: Int) {p, end, seg, cap, sp, cp, lp, at_eof}:
            cp[s] = 0
            lp[s] = 0
            var lo = s * seg
            var hi = min(lo + seg, end)
            if lo >= end:
                return
            var i = lo
            if s > 0:
                # move to a line start, then to the first plausible record start
                if p[i - 1] != 10:
                    while i < end and p[i] != 10:
                        i += 1
                    i += 1
                var found = False
                while i < hi + 4096 and i < end and not found:
                    if p[i] == 64:
                        var e0 = i
                        while e0 < end and p[e0] != 10:
                            e0 += 1
                        var s1 = e0 + 1
                        var e1 = s1
                        while e1 < end and p[e1] != 10:
                            e1 += 1
                        var s2 = e1 + 1
                        var e2 = s2
                        while e2 < end and p[e2] != 10:
                            e2 += 1
                        var s3 = e2 + 1
                        var e3 = s3
                        while e3 < end and p[e3] != 10:
                            e3 += 1
                        if s2 < end and p[s2] == 43 and e3 < end and (e1 - s1) == (e3 - s3):
                            found = True
                        else:
                            i = s1
                    else:
                        while i < end and p[i] != 10:
                            i += 1
                        i += 1
                if not found:
                    return
            var cnt = 0
            while i < hi and cnt < cap:
                if p[i] != 64:
                    break
                var e0 = i
                while e0 < end and p[e0] != 10:
                    e0 += 1
                var s1 = e0 + 1
                var e1 = s1
                while e1 < end and p[e1] != 10:
                    e1 += 1
                var s2 = e1 + 1
                var e2 = s2
                while e2 < end and p[e2] != 10:
                    e2 += 1
                var s3 = e2 + 1
                var e3 = s3 + (e1 - s1)  # quality line is as long as the sequence line
                if e3 > end or (e3 == end and not at_eof):
                    break  # incomplete record: leave it for the next block
                sp[s * cap + cnt] = Int32(i)
                cnt += 1
                i = e3 + 1
                lp[s] = i
            cp[s] = cnt

        sync_parallelize(scan, INDEX_SEGMENTS)
        var n = 0
        var last = 0
        for s in range(INDEX_SEGMENTS):
            var c = self.seg_counts[s]
            for k in range(c):
                self.starts[n + k] = self.seg_starts[s * cap + k]
            n += c
            if c > 0:
                last = self.seg_ends[s]
        self.n_rec = n
        self.cursor = 0
        # when everything is consumed, bytes after the last complete record are the carried tail
        self.block_end = last if n > 0 else 0
        if self.eof and (n == 0 or last >= end):
            self.block_end = end  # nothing more can be completed: discard any unparseable tail


def build_lut() -> List[UInt8]:
    var lut = List[UInt8](length=256, fill=4)
    for c in range(256):
        lut[c] = nt4_from_ascii(UInt8(c))
    return lut^


def read_templates(
    mut readers: List[FastqReader],
    paired: Bool,
    max_t: Int,
    lut: List[UInt8],
    h_reads: List[UInt8],
    h_rlen: List[Int32],
    names: List[UInt8],
    name_len: List[Int],
) raises -> Int:
    """Parse up to max_t templates into 4-bit slots (mate 1 at 2t, mate 2 at 2t+1 when paired).

    names holds NAME_STRIDE bytes per template (mate 1's name, '/1' stripped).
    """
    var t = 0
    var stride = 2 if paired else 1
    var lp = lut.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var slots = h_reads.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var rl = h_rlen.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var np = names.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var nl = name_len.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var err = List[Int](length=1, fill=0)
    var ep = err.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    while t < max_t:
        for r in range(len(readers)):
            while readers[r].cursor >= readers[r].n_rec and not readers[r].finished():
                readers[r].load_block()
        var avail = min(readers[0].n_rec - readers[0].cursor, max_t - t)
        if paired:
            avail = min(avail, readers[1].n_rec - readers[1].cursor)
        if avail <= 0:
            if paired and (readers[0].finished() != readers[1].finished()):
                raise Error("mate files have different numbers of reads (after " + String(t) + " templates)")
            break

        var b1 = readers[0].buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var s1 = readers[0].starts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + readers[0].cursor
        var b2 = b1
        var s2 = s1
        if paired:
            b2 = readers[1].buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            s2 = readers[1].starts.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + readers[1].cursor
        var t_base = t
        var tasks = min(PARSE_TASKS, avail)
        var per = (avail + tasks - 1) // tasks

        def parse(w: Int) {b1, s1, b2, s2, slots, rl, np, nl, ep, lp, t_base, per, avail, paired, stride}:
            var a = w * per
            var z = min(a + per, avail)
            for k in range(a, z):
                var tg = t_base + k
                for mate in range(2 if paired else 1):
                    var p = b1 if mate == 0 else b2
                    var o = Int((s1 if mate == 0 else s2)[k])
                    var i = o + 1  # skip '@'
                    var ns = i
                    while p[i] != 10 and p[i] != 32 and p[i] != 9:
                        i += 1
                    var ne = i
                    if mate == 0:
                        if ne - ns > 2 and p[ne - 2] == 47 and (p[ne - 1] == 49 or p[ne - 1] == 50):
                            ne -= 2
                        var nlen = ne - ns
                        if nlen > NAME_STRIDE:
                            ep[0] = 1
                            nlen = NAME_STRIDE
                        var dstn = np + tg * NAME_STRIDE
                        for q in range(nlen):
                            dstn[q] = p[ns + q]
                        nl[tg] = nlen
                    while p[i] != 10:
                        i += 1
                    i += 1
                    var slot = slots + (tg * stride + mate) * RSTRIDE
                    var L = 0
                    while True:
                        var c0 = p[i]
                        if c0 == 10 or c0 == 13:
                            break
                        var k0 = lp[Int(c0)]
                        i += 1
                        var c1 = p[i]
                        if c1 == 10 or c1 == 13:
                            if L < MAXL:
                                slot[L >> 1] = k0
                            L += 1
                            break
                        var k1 = lp[Int(c1)]
                        i += 1
                        if L < MAXL:
                            slot[L >> 1] = k0 | (k1 << 4)
                        L += 2
                    rl[tg * stride + mate] = Int32(L if L <= MAXL else 0)

        sync_parallelize(parse, tasks)
        if err[0] != 0:
            raise Error("read name longer than " + String(NAME_STRIDE) + " bytes")
        readers[0].cursor += avail
        if paired:
            readers[1].cursor += avail
        t += avail
    return t
