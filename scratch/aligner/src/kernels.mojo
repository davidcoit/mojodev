"""GPU splice-aware read alignment: seed -> chain -> breakpoint refinement.

One thread aligns one read.  Pipeline per read:
  1. minimizers of the read (same scanner as the reference index)
  2. look every minimizer up in the genome index -> anchors (ref, query, strand)
  3. insertion-sort anchors by (strand, ref), chain them with a DP whose gap
     cost distinguishes small indels (cheap, bounded) from introns (flat cost,
     up to MAX_INTRON)
  4. walk the best chain.  Where consecutive anchors sit on different
     diagonals, scan the unseeded gap for the best breakpoint (match count plus
     a bonus for GT..AG / CT..AC splice motifs) and emit I / D / N
  5. extend both ends with an ungapped X-drop and soft-clip the rest

Output is one fixed-size Int32 record per read; see OUT_* constants.
"""
from max.gpu import global_idx

from nt4 import PU8, PU32, PI32, nt4_get, nt4_complement
from minimizer import scan_minimizers, MM_K

comptime RSTRIDE = 128  # packed bytes per read slot -> reads up to 256 bases
comptime MAXL = 256
comptime MAXM = 128  # read minimizers kept
comptime MAXA = 256  # anchors kept per read
comptime MAX_OCC = 100  # skip minimizers more frequent than this in the genome
comptime MAX_BACK = 64  # chain DP look-back window (in sorted anchors)
comptime MAX_INTRON = 40000
comptime MAX_INDEL = 12
comptime MIN_INTRON = 20
comptime MIN_CHAIN_SCORE = 20
comptime XDROP = 8
comptime WIDEN = 8  # extra breakpoint slack on each side of the unseeded gap
comptime MIN_RESCUE = 8  # shortest overhang that end-rescue will try to place
comptime MAX_RESCUE_INTRON = 10000
comptime NONCANON_FLANK = 35  # a junction without a splice motif needs this much on both sides
comptime MIN_DB_OVERHANG = 3  # shortest overhang placed via a known junction

comptime OUT_STRIDE = 48  # ints per candidate alignment record
comptime NCAND = 2
comptime REC_STRIDE = OUT_STRIDE * NCAND
comptime OUT_OPS = 16  # ops start at this slot
comptime MAX_OPS = 30
# Record slots:
comptime O_STATUS = 0  # 0 unmapped, 1 mapped
comptime O_STRAND = 1
comptime O_POS = 2  # global 0-based start
comptime O_MAPQ = 3
comptime O_CHAIN = 4
comptime O_NM = 5
comptime O_NOPS = 6
comptime O_SECOND = 7
comptime O_SPAN = 8  # reference bases covered (incl. introns)
comptime O_NANCH = 9

comptime OP_M = 0
comptime OP_I = 1
comptime OP_D = 2
comptime OP_N = 3
comptime OP_S = 4


@always_inline
def g_get(g: PU8, n: Int, pos: Int) -> UInt8:
    if pos < 0 or pos >= n:
        return 4
    return nt4_get(g, pos)


@always_inline
def r_get(rd: PU8, L: Int, rev: Int, i: Int) -> UInt8:
    """Base i of the read in alignment orientation."""
    if i < 0 or i >= L:
        return 4
    if rev == 0:
        return nt4_get(rd, i)
    return nt4_complement(nt4_get(rd, L - 1 - i))


@always_inline
def is_match(a: UInt8, b: UInt8) -> Int:
    if a == b and a < 4:
        return 1
    return 0


def extend_right(g: PU8, n: Int, rd: PU8, L: Int, rev: Int, q0: Int, d: Int) -> Int:
    """Best ungapped extension length of query [q0, ...) on diagonal d."""
    var score = 0
    var best = 0
    var best_len = 0
    var q = q0
    while q < L:
        if is_match(r_get(rd, L, rev, q), g_get(g, n, q + d)) == 1:
            score += 1
        else:
            score -= 2
        q += 1
        if score > best:
            best = score
            best_len = q - q0
        elif score < best - XDROP:
            break
    return best_len


def extend_left(g: PU8, n: Int, rd: PU8, L: Int, rev: Int, q0: Int, d: Int) -> Int:
    """Best ungapped extension length of query (..., q0) going left on diagonal d."""
    var score = 0
    var best = 0
    var best_len = 0
    var q = q0 - 1
    while q >= 0:
        if is_match(r_get(rd, L, rev, q), g_get(g, n, q + d)) == 1:
            score += 1
        else:
            score -= 2
        if score > best:
            best = score
            best_len = q0 - q
        elif score < best - XDROP:
            break
        q -= 1
    return best_len


@always_inline
def push_op(ops: PI32, n_ops: Int, op: Int, length: Int) -> Int:
    """Append a CIGAR op (len << 3 | op), merging with the previous op if equal."""
    if length <= 0:
        return n_ops
    if n_ops > 0 and (Int(ops[n_ops - 1]) & 7) == op:
        ops[n_ops - 1] = Int32(((Int(ops[n_ops - 1]) >> 3) + length) << 3 | op)
        return n_ops
    if n_ops >= MAX_OPS:
        return n_ops + 1  # overflow marker, caller checks
    ops[n_ops] = Int32((length << 3) | op)
    return n_ops + 1


def count_matches(g: PU8, n: Int, rd: PU8, L: Int, rev: Int, q0: Int, q1: Int, d: Int) -> Int:
    var m = 0
    for q in range(q0, q1):
        m += is_match(r_get(rd, L, rev, q), g_get(g, n, q + d))
    return m


@always_inline
def splice_bonus(g: PU8, n: Int, intron_start: Int, intron_end: Int) -> Int:
    """+2 for a canonical GT..AG (forward) or CT..AC (reverse) intron [start, end)."""
    var d1 = g_get(g, n, intron_start)
    var d2 = g_get(g, n, intron_start + 1)
    var a1 = g_get(g, n, intron_end - 2)
    var a2 = g_get(g, n, intron_end - 1)
    if d1 == 2 and d2 == 3 and a1 == 0 and a2 == 2:
        return 2
    if d1 == 1 and d2 == 3 and a1 == 0 and a2 == 1:
        return 2
    return 0


def rescue_right(
    g: PU8, n: Int, rd: PU8, L: Int, rev: Int, d: Int, lo_b: Int, hi_b: Int
) -> Int:
    """Place the clipped right end of a read across a canonical intron.

    The exon is assumed to end at query b in [lo_b, hi_b] on diagonal d; the
    remaining read [b, L) must match (<=1 mismatch) right after an acceptor
    AG/AC downstream.  Returns (b << 32) | t, t = reference start of the next
    exon, or -1.
    """
    var best = -1
    var best_total = -1
    var baseline = count_matches(g, n, rd, L, rev, lo_b, L, d)
    var left = 0
    for b in range(lo_b, hi_b + 1):
        var r = L - b
        if r >= MIN_RESCUE:
            var s = b + d
            var d1 = g_get(g, n, s)
            var d2 = g_get(g, n, s + 1)
            var kind = 0
            if d1 == 2 and d2 == 3:
                kind = 1
            elif d1 == 1 and d2 == 3:
                kind = 2
            if kind != 0:
                var t = s + MIN_INTRON
                var tmax = min(s + MAX_RESCUE_INTRON, n - r)
                while t <= tmax:
                    var a1 = g_get(g, n, t - 2)
                    var a2 = g_get(g, n, t - 1)
                    if a1 == 0 and ((kind == 1 and a2 == 2) or (kind == 2 and a2 == 1)):
                        var mm = 0
                        var k = 0
                        while k < r and mm <= 1:
                            mm += 1 - is_match(r_get(rd, L, rev, b + k), g_get(g, n, t + k))
                            k += 1
                        if mm <= 1:
                            var total = left + (r - mm) + 2
                            if total > best_total:
                                best_total = total
                                best = (b << 32) | t
                    t += 1
        if b < hi_b:
            left += is_match(r_get(rd, L, rev, b), g_get(g, n, b + d))
    if best >= 0 and best_total >= baseline + 4:
        return best
    return -1


def rescue_left(
    g: PU8, n: Int, rd: PU8, L: Int, rev: Int, d: Int, lo_b: Int, hi_b: Int
) -> Int:
    """Mirror of rescue_right for a clipped left end.

    The main alignment starts at query b in [lo_b, hi_b] on diagonal d; read
    [0, b) must match (<=1 mismatch) just before a donor GT/CT upstream.
    Returns (b << 32) | s, s = reference start of the intron, or -1.
    """
    var best = -1
    var best_total = -1
    var baseline = count_matches(g, n, rd, L, rev, 0, hi_b, d)
    var main = 0
    var b = hi_b
    while b >= lo_b:
        if b < hi_b:
            main += is_match(r_get(rd, L, rev, b), g_get(g, n, b + d))
        if b >= MIN_RESCUE:
            var a = b + d
            var a1 = g_get(g, n, a - 2)
            var a2 = g_get(g, n, a - 1)
            var kind = 0
            if a1 == 0 and a2 == 2:
                kind = 1
            elif a1 == 0 and a2 == 1:
                kind = 2
            if kind != 0:
                var s = a - MIN_INTRON
                var smin = max(a - MAX_RESCUE_INTRON, b)
                while s >= smin:
                    var d1 = g_get(g, n, s)
                    var d2 = g_get(g, n, s + 1)
                    if d2 == 3 and ((kind == 1 and d1 == 2) or (kind == 2 and d1 == 1)):
                        var mm = 0
                        var k = 0
                        while k < b and mm <= 1:
                            mm += 1 - is_match(r_get(rd, L, rev, k), g_get(g, n, s - b + k))
                            k += 1
                        if mm <= 1:
                            var total = main + (b - mm) + 2
                            if total > best_total:
                                best_total = total
                                best = (b << 32) | s
                    s -= 1
        b -= 1
    if best >= 0 and best_total >= baseline + 4:
        return best
    return -1


@always_inline
def has_motif(g: PU8, n: Int, intron_start: Int, intron_end: Int) -> Bool:
    """Any known splice motif, either strand: GT..AG, GC..AG, AT..AC (CT..AC, CT..GC, GT..AT reversed)."""
    var d1 = g_get(g, n, intron_start)
    var d2 = g_get(g, n, intron_start + 1)
    var a1 = g_get(g, n, intron_end - 2)
    var a2 = g_get(g, n, intron_end - 1)
    if d1 == 2 and d2 == 3 and a1 == 0 and a2 == 2:
        return True
    if d1 == 1 and d2 == 3 and a1 == 0 and a2 == 1:
        return True
    if d1 == 2 and d2 == 1 and a1 == 0 and a2 == 2:
        return True
    if d1 == 1 and d2 == 3 and a1 == 2 and a2 == 1:
        return True
    if d1 == 0 and d2 == 3 and a1 == 0 and a2 == 1:
        return True
    if d1 == 2 and d2 == 3 and a1 == 0 and a2 == 3:
        return True
    return False


@always_inline
def lower_bound(arr: PU32, n: Int, key: Int) -> Int:
    """First index i with arr[i] >= key (arr sorted ascending)."""
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) >> 1
        if Int(arr[mid]) < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def db_right(
    g: PU8, n: Int, rd: PU8, L: Int, rev: Int, d: Int, lo_b: Int, hi_b: Int,
    nj: Int, js: PU32, je: PU32,
) -> Int:
    """Like rescue_right but only over known junctions (donor js[i] -> acceptor je[i])."""
    var best = -1
    var best_gain = 2
    for b in range(lo_b, hi_b + 1):
        var r = L - b
        if r >= MIN_DB_OVERHANG:
            var s = b + d
            var i = lower_bound(js, nj, s)
            var base_r = count_matches(g, n, rd, L, rev, b, L, d)
            while i < nj and Int(js[i]) == s:
                var t = Int(je[i])
                var allow = r // 10
                var mm = 0
                var k = 0
                while k < r and mm <= allow:
                    mm += 1 - is_match(r_get(rd, L, rev, b + k), g_get(g, n, t + k))
                    k += 1
                if mm <= allow:
                    var gain = (r - mm) - base_r - 2 * mm
                    if gain > best_gain:
                        best_gain = gain
                        best = (b << 32) | t
                i += 1
    return best


def db_left(
    g: PU8, n: Int, rd: PU8, L: Int, rev: Int, d: Int, lo_b: Int, hi_b: Int,
    nj: Int, ae: PU32, a_s: PU32,
) -> Int:
    """Like rescue_left but only over known junctions (acceptor ae[i] <- donor a_s[i])."""
    var best = -1
    var best_gain = 2
    var b = hi_b
    while b >= lo_b:
        if b >= MIN_DB_OVERHANG:
            var a = b + d
            var i = lower_bound(ae, nj, a)
            var base_l = count_matches(g, n, rd, L, rev, 0, b, d)
            while i < nj and Int(ae[i]) == a:
                var sp = Int(a_s[i])
                var allow = b // 10
                var mm = 0
                var k = 0
                while k < b and mm <= allow:
                    mm += 1 - is_match(r_get(rd, L, rev, k), g_get(g, n, sp - b + k))
                    k += 1
                if mm <= allow:
                    var gain = (b - mm) - base_l - 2 * mm
                    if gain > best_gain:
                        best_gain = gain
                        best = (b << 32) | sp
                i += 1
        b -= 1
    return best


def build_alignment(
    genome: PU8,
    n: Int,
    rd: PU8,
    L: Int,
    rev: Int,
    ak: PU32,
    aq: PI32,
    ac: PI32,
    m: Int,
    best_f: Int,
    second: Int,
    n_junc: Int,
    j_donor: PU32,
    j_acc_of_donor: PU32,
    j_acc: PU32,
    j_donor_of_acc: PU32,
    rec: PI32,
):
    """Turn a chain (anchor indices ac[0:m], ascending) into a CIGAR record."""
    var ops = rec + OUT_OPS
    var n_ops = 0
    var first = Int(ac[0])
    var r_first = Int(ak[first] & 0x7FFFFFFF)
    var q_first = Int(aq[first])
    var d_cur = r_first - q_first

    var le = extend_left(genome, n, rd, L, rev, q_first, d_cur)
    var qs = q_first - le
    var ref_start = qs + d_cur
    var emit_q = qs  # next query base to emit
    var cov_q = q_first + MM_K  # end of what the current diagonal certainly covers
    var nm = 0
    var ref_span = 0
    var rescued = 0
    var seg_q = qs  # query start of the current exon
    var n_splices = 0
    # Clipped / over-extended left end: try known junctions, then a de novo
    # canonical-intron search, to place the first exon.
    var rl = -1
    var hi_l = min(qs + 8, q_first)
    if n_junc > 0:
        var lo_db = max(MIN_DB_OVERHANG, qs - 8)
        if lo_db <= hi_l:
            rl = db_left(genome, n, rd, L, rev, d_cur, lo_db, hi_l, n_junc, j_acc, j_donor_of_acc)
    if rl < 0 and qs >= 4:
        var lo_dn = max(MIN_RESCUE, qs - 4)
        if lo_dn <= hi_l:
            rl = rescue_left(genome, n, rd, L, rev, d_cur, lo_dn, hi_l)
    if rl >= 0:
        var bq = rl >> 32
        var sp = rl & 0xFFFFFFFF
        ref_start = sp - bq
        nm += bq - count_matches(genome, n, rd, L, rev, 0, bq, sp - bq)
        n_ops = push_op(ops, n_ops, OP_M, bq)
        n_ops = push_op(ops, n_ops, OP_N, (bq + d_cur) - sp)
        ref_span += bq + (bq + d_cur) - sp
        emit_q = bq
        seg_q = bq
        n_splices = 1
        rescued = 1
    if rescued == 0:
        n_ops = push_op(ops, n_ops, OP_S, qs)

    for t in range(1, m):
        var ai = Int(ac[t])
        var rt = Int(ak[ai] & 0x7FFFFFFF)
        var qt = Int(aq[ai])
        var dt = rt - qt
        if dt == d_cur:
            cov_q = max(cov_q, qt + MM_K)
            continue
        var delta = dt - d_cur
        var ins = 0
        if delta < 0:
            ins = -delta
        # breakpoint search window on the query
        var b_lo = min(cov_q, qt) - WIDEN
        var b_hi = min(max(cov_q, qt) + WIDEN, qt + MM_K)
        if ins > 0:
            b_hi = qt - ins
            b_lo = min(cov_q, b_hi) - WIDEN
        b_lo = max(b_lo, emit_q)
        if b_hi < b_lo:
            b_hi = b_lo
        # right-hand matches over the whole window, then slide the breakpoint
        var right_total = count_matches(genome, n, rd, L, rev, b_lo + ins, b_hi + ins, dt)
        var left = 0
        var right_used = 0
        var best_b = b_lo
        var best_s = -1000
        for b in range(b_lo, b_hi + 1):
            var s = left + (right_total - right_used)
            if delta >= MIN_INTRON:
                s += splice_bonus(genome, n, b + d_cur, b + dt)
            if s > best_s:
                best_s = s
                best_b = b
            if b < b_hi:
                left += is_match(r_get(rd, L, rev, b), g_get(genome, n, b + d_cur))
                right_used += is_match(r_get(rd, L, rev, b + ins), g_get(genome, n, b + ins + dt))
        # An intron with no splice motif is only believed when both flanks are long.
        # Otherwise drop the short side: truncate the right tail (the end rescue may
        # still re-place it across a canonical intron) or clip a short first exon.
        if delta >= MIN_INTRON and not has_motif(genome, n, best_b + d_cur, best_b + dt):
            var left_flank = best_b - seg_q
            var right_flank = L - best_b
            if min(left_flank, right_flank) < NONCANON_FLANK:
                if right_flank <= left_flank:
                    break
                if n_splices == 0:
                    var rs0 = best_b + ins
                    n_ops = 0
                    nm = 0
                    ref_span = 0
                    n_ops = push_op(ops, n_ops, OP_S, rs0)
                    ref_start = rs0 + dt
                    emit_q = rs0
                    seg_q = rs0
                    d_cur = dt
                    cov_q = max(rs0, qt + MM_K)
                    continue
        # emit M up to the breakpoint, then the gap op
        nm += (best_b - emit_q) - count_matches(genome, n, rd, L, rev, emit_q, best_b, d_cur)
        n_ops = push_op(ops, n_ops, OP_M, best_b - emit_q)
        ref_span += best_b - emit_q
        if delta < 0:
            n_ops = push_op(ops, n_ops, OP_I, ins)
            nm += ins
        elif delta < MIN_INTRON:
            n_ops = push_op(ops, n_ops, OP_D, delta)
            nm += delta
            ref_span += delta
        else:
            n_ops = push_op(ops, n_ops, OP_N, delta)
            ref_span += delta
            seg_q = best_b + ins
            n_splices += 1
        emit_q = best_b + ins
        d_cur = dt
        cov_q = max(emit_q, qt + MM_K)

    var re = extend_right(genome, n, rd, L, rev, cov_q, d_cur)
    var q_end = min(cov_q + re, L)
    var rr = -1
    var lo_r = max(max(q_end - 8, cov_q - 3), emit_q + 1)
    if n_junc > 0:
        var hi_db = min(q_end + 4, L - MIN_DB_OVERHANG)
        if lo_r <= hi_db:
            rr = db_right(genome, n, rd, L, rev, d_cur, lo_r, hi_db, n_junc, j_donor, j_acc_of_donor)
    if rr < 0 and L - q_end >= 4:
        var hi = min(q_end + 4, L - MIN_RESCUE)
        if lo_r <= hi:
            rr = rescue_right(genome, n, rd, L, rev, d_cur, lo_r, hi)
    if rr >= 0:
        var bq = rr >> 32
        var tp = rr & 0xFFFFFFFF
        nm += (bq - emit_q) - count_matches(genome, n, rd, L, rev, emit_q, bq, d_cur)
        n_ops = push_op(ops, n_ops, OP_M, bq - emit_q)
        n_ops = push_op(ops, n_ops, OP_N, tp - (bq + d_cur))
        nm += (L - bq) - count_matches(genome, n, rd, L, rev, bq, L, tp - bq)
        n_ops = push_op(ops, n_ops, OP_M, L - bq)
        ref_span += (bq - emit_q) + (tp - (bq + d_cur)) + (L - bq)
    else:
        nm += (q_end - emit_q) - count_matches(genome, n, rd, L, rev, emit_q, q_end, d_cur)
        n_ops = push_op(ops, n_ops, OP_M, q_end - emit_q)
        ref_span += q_end - emit_q
        n_ops = push_op(ops, n_ops, OP_S, L - q_end)
    if n_ops > MAX_OPS:
        return

    var mapq = 60
    if second > 0:
        var ratio = (second * 100) // best_f
        if ratio >= 95:
            mapq = 0
        elif ratio >= 80:
            mapq = 1
        elif ratio >= 60:
            mapq = 10
        else:
            mapq = 40
    rec[O_STATUS] = 1
    rec[O_STRAND] = Int32(rev)
    rec[O_POS] = Int32(ref_start)
    rec[O_MAPQ] = Int32(mapq)
    rec[O_CHAIN] = Int32(best_f)
    rec[O_NM] = Int32(nm)
    rec[O_NOPS] = Int32(n_ops)
    rec[O_SECOND] = Int32(second)
    rec[O_SPAN] = Int32(ref_span)


def align_kernel(
    genome: PU8,
    n_genome: Int32,
    bucket_off: PU32,
    ent_hash: PU32,
    ent_pos: PU32,
    reads: PU8,
    rlen: PI32,
    n_reads: Int32,
    base: Int32,
    count: Int32,
    n_junc: Int32,
    j_donor: PU32,
    j_acc_of_donor: PU32,
    j_acc: PU32,
    j_donor_of_acc: PU32,
    m_hash: PU32,
    m_pos: PU32,
    a_key: PU32,
    a_q: PI32,
    a_f: PI32,
    a_prev: PI32,
    a_chain: PI32,
    out_buf: PI32,
):
    var lid = Int(global_idx.x)  # slot in this batch (scratch is per slot)
    if lid >= Int(count):  # grid is rounded up to a block multiple: stay inside this batch
        return
    var rid = lid + Int(base)  # global read index (reads / output are global)
    if rid >= Int(n_reads):
        return
    var rec = out_buf + rid * REC_STRIDE
    for i in range(REC_STRIDE):
        rec[i] = 0
    var L = Int(rlen[rid])
    if L < MM_K + 8 or L > MAXL:
        return
    var n = Int(n_genome)
    var rd = reads + rid * RSTRIDE
    var mh = m_hash + lid * MAXM
    var mp = m_pos + lid * MAXM
    var ak = a_key + lid * MAXA
    var aq = a_q + lid * MAXA
    var af = a_f + lid * MAXA
    var ap = a_prev + lid * MAXA
    var ac = a_chain + lid * MAXA

    # ---- 1/2: seeds -> anchors
    var n_min = scan_minimizers(rd, 0, L, mh, mp, MAXM)
    var n_a = 0
    for mi in range(n_min):
        var h = mh[mi]
        var qpos = Int(mp[mi] & 0x7FFFFFFF)
        var rs = Int(mp[mi] >> 31)
        var b = Int(h >> 8)
        var lo = Int(bucket_off[b])
        var hi = Int(bucket_off[b + 1])
        var cnt = 0
        for e in range(lo, hi):
            if ent_hash[e] == h:
                cnt += 1
        if cnt == 0 or cnt > MAX_OCC:
            continue
        for e in range(lo, hi):
            if ent_hash[e] != h:
                continue
            if n_a >= MAXA:
                break
            var gs = Int(ent_pos[e] >> 31)
            var gpos = Int(ent_pos[e] & 0x7FFFFFFF)
            var rev = 0
            if gs != rs:
                rev = 1
            var q = qpos
            if rev == 1:
                q = L - qpos - MM_K
            # insertion sort by key = rev << 31 | gpos
            var key = UInt32((rev << 31) | gpos)
            var j = n_a
            while j > 0 and ak[j - 1] > key:
                ak[j] = ak[j - 1]
                aq[j] = aq[j - 1]
                j -= 1
            ak[j] = key
            aq[j] = Int32(q)
            n_a += 1
    rec[O_NANCH] = Int32(n_a)
    if n_a == 0:
        return

    # ---- 3: chaining DP over anchors sorted by (strand, ref)
    var best_i = 0
    var best_f = 0
    for i in range(n_a):
        var ri = Int(ak[i] & 0x7FFFFFFF)
        var si = Int(ak[i] >> 31)
        var qi = Int(aq[i])
        var f = MM_K
        var pv = -1
        var jmin = max(0, i - MAX_BACK)
        var j = i - 1
        while j >= jmin:
            if Int(ak[j] >> 31) != si:
                break
            var dr = ri - Int(ak[j] & 0x7FFFFFFF)
            if dr > MAX_INTRON + MAXL:
                break
            var dq = qi - Int(aq[j])
            if dr > 0 and dq > 0:
                var dd = dr - dq
                var cost = -1
                if dd == 0:
                    cost = 0
                elif dd < 0 and -dd <= MAX_INDEL:
                    cost = 3 - dd
                elif dd > 0 and dd <= MAX_INDEL:
                    cost = 3 + dd
                elif dd >= MIN_INTRON and dd <= MAX_INTRON:
                    cost = 8
                if cost >= 0:
                    var gain = min(min(dq, dr), MM_K)
                    var sc = Int(af[j]) + gain - cost
                    if sc > f:
                        f = sc
                        pv = j
            j -= 1
        af[i] = Int32(f)
        ap[i] = Int32(pv)
        if f > best_f:
            best_f = f
            best_i = i
    if best_f < MIN_CHAIN_SCORE:
        return

    # backtrack the best chain, then reverse into ascending order
    var m = 0
    var cur = best_i
    while cur >= 0 and m < MAXA:
        ac[m] = Int32(cur)
        m += 1
        cur = Int(ap[cur])
    for t in range(m // 2):
        var tmp = ac[t]
        ac[t] = ac[m - 1 - t]
        ac[m - 1 - t] = tmp

    # runner-up chain at a different locus: best f among anchors outside the best
    # chain's own reference span, so tandem duplicates count as competing loci
    var best_strand = Int(ak[best_i] >> 31)
    var span_lo = Int(ak[Int(ac[0])] & 0x7FFFFFFF) - 2 * MM_K
    var span_hi = Int(ak[Int(ac[m - 1])] & 0x7FFFFFFF) + 3 * MM_K
    var second = 0
    var second_i = -1
    for i in range(n_a):
        var inside = 0
        if Int(ak[i] >> 31) == best_strand:
            var ri2 = Int(ak[i] & 0x7FFFFFFF)
            if ri2 >= span_lo and ri2 <= span_hi:
                inside = 1
        if inside == 0 and Int(af[i]) > second:
            second = Int(af[i])
            second_i = i

    build_alignment(genome, n, rd, L, best_strand, ak, aq, ac, m, best_f, second, Int(n_junc), j_donor, j_acc_of_donor, j_acc, j_donor_of_acc, rec)

    # candidate 1: the runner-up locus, if it is a credible alignment
    if second_i >= 0 and second >= MIN_CHAIN_SCORE and 2 * second >= best_f:
        var m2 = 0
        var cur2 = second_i
        while cur2 >= 0 and m2 < MAXA:
            var in_span = 0
            if Int(ak[cur2] >> 31) == best_strand:
                var r3 = Int(ak[cur2] & 0x7FFFFFFF)
                if r3 >= span_lo and r3 <= span_hi:
                    in_span = 1
            if in_span == 1:
                break
            ac[m2] = Int32(cur2)
            m2 += 1
            cur2 = Int(ap[cur2])
        for t in range(m2 // 2):
            var tmp2 = ac[t]
            ac[t] = ac[m2 - 1 - t]
            ac[m2 - 1 - t] = tmp2
        if m2 > 0:
            var rec2 = rec + OUT_STRIDE
            for i in range(OUT_STRIDE):
                rec2[i] = 0
            build_alignment(genome, n, rd, L, Int(ak[second_i] >> 31), ak, aq, ac, m2, second, best_f, Int(n_junc), j_donor, j_acc_of_donor, j_acc, j_donor_of_acc, rec2)
