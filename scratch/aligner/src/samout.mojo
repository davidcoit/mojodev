"""Pair resolution and SAM output (host side).

Each read arrives with up to NCAND candidate alignments from the GPU.  For a pair we
look at every combination of one candidate per mate, score the compatible ones
(opposite strands, same chromosome, forward mate not past the end of the reverse mate,
bounded genomic span) with a bonus, and keep the best.  The runner-up combination
sets the pair-level MAPQ, so a mate that is ambiguous on its own can be resolved by
its partner.
"""
from index import Reference
from kernels import REC_STRIDE, OUT_STRIDE, OUT_OPS, O_STATUS, O_STRAND, O_POS, O_MAPQ, O_CHAIN, O_NM, O_NOPS, O_SPAN, OP_N

comptime PAIR_BONUS = 20


def put_str(mut buf: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(len(b)):
        buf.append(b[i])


def put_int(mut buf: List[UInt8], v: Int):
    if v == 0:
        buf.append(48)
        return
    var x = v
    if x < 0:
        buf.append(45)
        x = -x
    var start = len(buf)
    while x > 0:
        buf.append(UInt8(48 + x % 10))
        x //= 10
    var i = start
    var j = len(buf) - 1
    while i < j:
        var t = buf[i]
        buf[i] = buf[j]
        buf[j] = t
        i += 1
        j -= 1


struct Cand(Copyable, Movable):
    var valid: Bool
    var chrom: Int
    var strand: Int
    var gpos: Int  # global 0-based start
    var end: Int  # gpos + reference span (exclusive)
    var mapq: Int
    var chain: Int
    var nm: Int
    var ascore: Int  # alignment score: aligned bases - 3 * edits (match +1, mismatch -2)
    var off: Int  # offset of this candidate's record in the record list

    def __init__(out self):
        self.valid = False
        self.chrom = 0
        self.strand = 0
        self.gpos = 0
        self.end = 0
        self.mapq = 0
        self.chain = 0
        self.nm = 0
        self.ascore = 0
        self.off = 0

    def __init__(out self, chrom: Int, strand: Int, gpos: Int, end: Int, mapq: Int, chain: Int, nm: Int, ascore: Int, off: Int):
        self.valid = True
        self.chrom = chrom
        self.strand = strand
        self.gpos = gpos
        self.end = end
        self.mapq = mapq
        self.chain = chain
        self.nm = nm
        self.ascore = ascore
        self.off = off


struct Stats(Movable):
    var templates: Int
    var mate1_mapped: Int
    var mate2_mapped: Int
    var proper: Int
    var both_mapped: Int
    var one_mapped: Int
    var unmapped: Int
    var spliced_mates: Int
    var reads_mapped: Int
    var reads_total: Int

    def __init__(out self):
        self.templates = 0
        self.mate1_mapped = 0
        self.mate2_mapped = 0
        self.proper = 0
        self.both_mapped = 0
        self.one_mapped = 0
        self.unmapped = 0
        self.spliced_mates = 0
        self.reads_mapped = 0
        self.reads_total = 0

    def merge(mut self, o: Stats):
        self.templates += o.templates
        self.mate1_mapped += o.mate1_mapped
        self.mate2_mapped += o.mate2_mapped
        self.proper += o.proper
        self.both_mapped += o.both_mapped
        self.one_mapped += o.one_mapped
        self.unmapped += o.unmapped
        self.spliced_mates += o.spliced_mates
        self.reads_mapped += o.reads_mapped
        self.reads_total += o.reads_total


def load_cand(recs: List[Int32], slot: Int, c: Int, ref_: Reference) -> Cand:
    var off = slot * REC_STRIDE + c * OUT_STRIDE
    var p = recs.unsafe_ptr() + off
    if Int(p[O_STATUS]) != 1:
        return Cand()
    var gpos = Int(p[O_POS])
    var span = Int(p[O_SPAN])
    var chrom = ref_.chrom_of(gpos)
    if gpos + span > ref_.starts[chrom] + ref_.lengths[chrom]:
        return Cand()
    var msum = 0
    for t in range(Int(p[O_NOPS])):
        var v = Int(p[OUT_OPS + t])
        if (v & 7) == 0:
            msum += v >> 3
    var nm = Int(p[O_NM])
    return Cand(chrom, Int(p[O_STRAND]), gpos, gpos + span, Int(p[O_MAPQ]), Int(p[O_CHAIN]), nm, msum - 3 * nm, off)


def compatible(a: Cand, b: Cand, max_frag: Int) -> Bool:
    if not a.valid or not b.valid:
        return False
    if a.chrom != b.chrom or a.strand == b.strand:
        return False
    var f_start = a.gpos
    var r_end = b.end
    if a.strand == 1:  # a is the reverse mate
        f_start = b.gpos
        r_end = a.end
    if f_start > r_end:
        return False
    var lo = min(a.gpos, b.gpos)
    var hi = max(a.end, b.end)
    return hi - lo <= max_frag


def ratio_mapq(best: Int, second: Int) -> Int:
    if second <= 0:
        return 60
    var ratio = (second * 100) // max(best, 1)
    if ratio >= 95:
        return 0
    if ratio >= 80:
        return 1
    if ratio >= 60:
        return 10
    return 40


def put_record(
    mut sam: List[UInt8],
    names: List[UInt8],
    name_off: Int,
    name_len: Int,
    flag: Int,
    rname: String,
    pos1: Int,
    mapq: Int,
    recs: List[Int32],
    c: Cand,
    rnext: String,
    pnext: Int,
    tlen: Int,
    nh: Int,
):
    for i in range(name_len):
        sam.append(names[name_off + i])
    sam.append(9)
    put_int(sam, flag)
    sam.append(9)
    put_str(sam, rname)
    sam.append(9)
    put_int(sam, pos1)
    sam.append(9)
    put_int(sam, mapq)
    sam.append(9)
    if c.valid:
        var p = recs.unsafe_ptr() + c.off
        for t in range(Int(p[O_NOPS])):
            var v = Int(p[OUT_OPS + t])
            put_int(sam, v >> 3)
            var code = v & 7
            var ch = UInt8(83)  # S
            if code == 0:
                ch = 77  # M
            elif code == 1:
                ch = 73  # I
            elif code == 2:
                ch = 68  # D
            elif code == 3:
                ch = 78  # N
            sam.append(ch)
    else:
        sam.append(42)
    sam.append(9)
    put_str(sam, rnext)
    sam.append(9)
    put_int(sam, pnext)
    sam.append(9)
    put_int(sam, tlen)
    put_str(sam, "\t*\t*")
    if c.valid:
        put_str(sam, "\tNM:i:")
        put_int(sam, c.nm)
        put_str(sam, "\tAS:i:")
        put_int(sam, c.chain)
        put_str(sam, "\tNH:i:")
        put_int(sam, nh)
        put_str(sam, "\tHI:i:1")
    sam.append(10)


def has_splice(recs: List[Int32], c: Cand) -> Bool:
    if not c.valid:
        return False
    var p = recs.unsafe_ptr() + c.off
    for t in range(Int(p[O_NOPS])):
        if (Int(p[OUT_OPS + t]) & 7) == OP_N:
            return True
    return False


def write_template(
    mut sam: List[UInt8],
    recs: List[Int32],
    t: Int,
    paired: Bool,
    max_frag: Int,
    ref_: Reference,
    names: List[UInt8],
    name_off: Int,
    name_len: Int,
    mut stats: Stats,
):
    """Resolve template t and write its records.

    Paired: mate 1 is in slot 2t and mate 2 in slot 2t + 1.  Single-end: slot t.
    """
    var s1 = 2 * t if paired else t
    var a0 = load_cand(recs, s1, 0, ref_)
    var a1 = load_cand(recs, s1, 1, ref_)
    var b0 = Cand()
    var b1 = Cand()
    if paired:
        b0 = load_cand(recs, s1 + 1, 0, ref_)
        b1 = load_cand(recs, s1 + 1, 1, ref_)

    var ca = a0.copy()
    var cb = b0.copy()
    var ia = 0  # index of the chosen candidate of each mate
    var ib = 0
    var proper = False
    var pair_mapq = 0
    if paired:
        var best = -1
        var runner = -1
        var bi = 0
        var bj = 0
        for i in range(2):
            for j in range(2):
                var ci = a0.copy() if i == 0 else a1.copy()
                var cj = b0.copy() if j == 0 else b1.copy()
                if not (ci.valid and cj.valid):
                    continue
                var sc = ci.chain + cj.chain
                if compatible(ci, cj, max_frag):
                    sc += PAIR_BONUS
                if sc > best:
                    runner = best
                    best = sc
                    bi = i
                    bj = j
                elif sc > runner:
                    runner = sc
        if best >= 0:
            ia = bi
            ib = bj
            ca = a0.copy() if bi == 0 else a1.copy()
            cb = b0.copy() if bj == 0 else b1.copy()
            proper = compatible(ca, cb, max_frag)
            if proper:
                pair_mapq = ratio_mapq(best, runner)

    stats.templates += 1
    var n_mates = 2 if paired else 1
    for m in range(n_mates):
        var me = ca.copy() if m == 0 else cb.copy()
        var mate = cb.copy() if m == 0 else ca.copy()
        if not paired:
            mate = Cand()
        stats.reads_total += 1
        if me.valid:
            stats.reads_mapped += 1
            if has_splice(recs, me):
                stats.spliced_mates += 1
        var flag = 0
        if paired:
            flag = 1 | (64 if m == 0 else 128)
            if proper:
                flag |= 2
            if not me.valid:
                flag |= 4
            if not mate.valid:
                flag |= 8
            if me.valid and me.strand == 1:
                flag |= 16
            if mate.valid and mate.strand == 1:
                flag |= 32
        else:
            if not me.valid:
                flag = 4
            elif me.strand == 1:
                flag = 16
        # NH: 2 when another locus has an alignment score within 1 of the chosen one (STAR's
        # outFilterMultimapScoreRange), unless the partner resolved the ambiguity (proper
        # pair with a confident pair MAPQ)
        var other = Cand()
        if m == 0:
            other = a1.copy() if ia == 0 else a0.copy()
        else:
            other = b1.copy() if ib == 0 else b0.copy()
        var nh = 1
        if me.valid and other.valid and other.ascore >= me.ascore - 1:
            nh = 2
            if proper and pair_mapq >= 10:
                nh = 1
        var mapq = me.mapq
        if proper:
            mapq = max(mapq, pair_mapq)
        if not me.valid:
            mapq = 0
        var rname = String("*")
        var pos1 = 0
        if me.valid:
            rname = ref_.names[me.chrom]
            pos1 = me.gpos - ref_.starts[me.chrom] + 1
        elif mate.valid:
            rname = ref_.names[mate.chrom]
            pos1 = mate.gpos - ref_.starts[mate.chrom] + 1
        var rnext = String("*")
        var pnext = 0
        var tlen = 0
        if mate.valid:
            rnext = String("=") if (me.valid and me.chrom == mate.chrom) or not me.valid else ref_.names[mate.chrom]
            pnext = mate.gpos - ref_.starts[mate.chrom] + 1
            if me.valid and me.chrom == mate.chrom:
                var lo = min(me.gpos, mate.gpos)
                var hi = max(me.end, mate.end)
                tlen = hi - lo
                if me.gpos > mate.gpos or (me.gpos == mate.gpos and m == 1):
                    tlen = -tlen
        elif me.valid and paired:
            rnext = String("=")
            pnext = pos1
        put_record(sam, names, name_off, name_len, flag, rname, pos1, mapq, recs, me, rnext, pnext, tlen, nh)

    if paired:
        if ca.valid:
            stats.mate1_mapped += 1
        if cb.valid:
            stats.mate2_mapped += 1
        if ca.valid and cb.valid:
            stats.both_mapped += 1
            if proper:
                stats.proper += 1
        elif ca.valid or cb.valid:
            stats.one_mapped += 1
        else:
            stats.unmapped += 1
    else:
        if ca.valid:
            stats.mate1_mapped += 1
        else:
            stats.unmapped += 1
