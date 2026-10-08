"""GPU splice-aware aligner.

Usage: main REF.fa READS.fastq OUT.sam [batch_reads=200000] [passes=2] [min_junction_support=1]

Pass 1 aligns every read de novo.  Pass 2 (default) re-aligns with a table of the
junctions pass 1 found, which lets reads with very short overhangs be placed.
"""
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from nt4 import PU8, nt4_from_ascii, nt4_set, nt4_bytes
from index import load_fasta, build_index
from junctions import collect_junctions
from kernels import (
    align_kernel,
    RSTRIDE,
    MAXL,
    MAXM,
    MAXA,
    OUT_STRIDE,
    OUT_OPS,
    O_STATUS,
    O_STRAND,
    O_POS,
    O_MAPQ,
    O_CHAIN,
    O_NM,
    O_NOPS,
    O_SPAN,
)

comptime BLOCK = 128


def secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


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
    # digits were appended least-significant first: reverse in place
    var i = start
    var j = len(buf) - 1
    while i < j:
        var t = buf[i]
        buf[i] = buf[j]
        buf[j] = t
        i += 1
        j -= 1


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("usage: main REF.fa READS.fastq OUT.sam [batch_reads] [passes=2] [min_junction_support=1]")
        return
    var batch = 200000
    if len(args) > 4:
        batch = Int(String(args[4]))
    var n_passes = 2
    if len(args) > 5:
        n_passes = Int(String(args[5]))
    var min_support = 1
    if len(args) > 6:
        min_support = Int(String(args[6]))

    var ctx = DeviceContext()
    print("device:", ctx.name())

    # ---- reference + index
    var t0 = perf_counter_ns()
    var ref_ = load_fasta(String(args[1]))
    print("reference:", len(ref_.names), "sequences,", ref_.n_bases, "bases (", secs(t0), "s )")
    var idx = build_index(ref_)

    var d_genome = ctx.enqueue_create_buffer[DType.uint8](nt4_bytes(ref_.n_bases) + 1)
    var d_bucket = ctx.enqueue_create_buffer[DType.uint32](len(idx.bucket_off))
    var d_ehash = ctx.enqueue_create_buffer[DType.uint32](idx.n_entries)
    var d_epos = ctx.enqueue_create_buffer[DType.uint32](idx.n_entries)
    ctx.enqueue_copy(d_genome, ref_.packed.unsafe_ptr())
    ctx.enqueue_copy(d_bucket, idx.bucket_off.unsafe_ptr())
    ctx.enqueue_copy(d_ehash, idx.ent_hash.unsafe_ptr())
    ctx.enqueue_copy(d_epos, idx.ent_pos.unsafe_ptr())
    ctx.synchronize()
    var index_mb = Float64(len(idx.bucket_off) * 4 + idx.n_entries * 8 + nt4_bytes(ref_.n_bases)) / 1e6
    print("device index:", index_mb, "MB")

    # ---- reads -> 4-bit packed slots
    t0 = perf_counter_ns()
    var fq = open(String(args[2]), "r")
    var size = Int(fq.seek(0, 2))
    _ = fq.seek(0, 0)
    var data = List[UInt8](length=size, fill=0)
    _ = fq.read(Span(data))
    var dp = data.unsafe_ptr()
    print("  fastq read:", secs(t0), "s")
    var count = 0
    for i in range(size):
        if dp[i] == 10:
            count += 1
    var n_reads = count // 4
    var h_reads = List[UInt8](length=n_reads * RSTRIDE, fill=0)
    var h_rlen = List[Int32](length=n_reads, fill=0)
    var name_start = List[Int](length=n_reads, fill=0)
    var name_len = List[Int](length=n_reads, fill=0)
    var rp = h_reads.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pos = 0
    for r in range(n_reads):
        pos += 1  # '@'
        name_start[r] = pos
        while dp[pos] != 10 and dp[pos] != 32:
            pos += 1
        name_len[r] = pos - name_start[r]
        while dp[pos] != 10:
            pos += 1
        pos += 1
        var L = 0
        var slot = rp + r * RSTRIDE
        while dp[pos] != 10:
            if L < MAXL:
                nt4_set(slot, L, nt4_from_ascii(dp[pos]))
            L += 1
            pos += 1
        pos += 1
        while dp[pos] != 10:  # '+'
            pos += 1
        pos += 1
        pos += L  # qualities (same length as the sequence)
        while dp[pos] != 10:
            pos += 1
        pos += 1
        if L > MAXL:
            L = 0  # too long: left unmapped
        h_rlen[r] = Int32(L)
    print("reads:", n_reads, "parsed in", secs(t0), "s")

    var d_reads = ctx.enqueue_create_buffer[DType.uint8](n_reads * RSTRIDE)
    var d_rlen = ctx.enqueue_create_buffer[DType.int32](n_reads)
    var d_out = ctx.enqueue_create_buffer[DType.int32](n_reads * OUT_STRIDE)
    var n_slots = min(batch, n_reads)
    var d_mh = ctx.enqueue_create_buffer[DType.uint32](n_slots * MAXM)
    var d_mp = ctx.enqueue_create_buffer[DType.uint32](n_slots * MAXM)
    var d_ak = ctx.enqueue_create_buffer[DType.uint32](n_slots * MAXA)
    var d_aq = ctx.enqueue_create_buffer[DType.int32](n_slots * MAXA)
    var d_af = ctx.enqueue_create_buffer[DType.int32](n_slots * MAXA)
    var d_ap = ctx.enqueue_create_buffer[DType.int32](n_slots * MAXA)
    var d_ac = ctx.enqueue_create_buffer[DType.int32](n_slots * MAXA)
    ctx.enqueue_copy(d_reads, h_reads.unsafe_ptr())
    ctx.enqueue_copy(d_rlen, h_rlen.unsafe_ptr())
    ctx.synchronize()

    # ---- align in batches; pass 2 re-aligns with a junction table from pass 1
    var h_out = List[Int32](length=n_reads * OUT_STRIDE, fill=0)
    var n_junc = 0
    var d_jd = ctx.enqueue_create_buffer[DType.uint32](1)
    var d_jda = ctx.enqueue_create_buffer[DType.uint32](1)
    var d_ja = ctx.enqueue_create_buffer[DType.uint32](1)
    var d_jad = ctx.enqueue_create_buffer[DType.uint32](1)
    for pass_ in range(n_passes):
        if pass_ == 1:
            ctx.enqueue_copy(h_out.unsafe_ptr(), d_out)
            ctx.synchronize()
            t0 = perf_counter_ns()
            var jdb = collect_junctions(h_out, n_reads, min_support, 12)
            n_junc = jdb.n
            print("pass 1 junctions with >=", min_support, "supporting reads:", n_junc, "(", secs(t0), "s on host )")
            d_jd = ctx.enqueue_create_buffer[DType.uint32](len(jdb.donor))
            d_jda = ctx.enqueue_create_buffer[DType.uint32](len(jdb.donor))
            d_ja = ctx.enqueue_create_buffer[DType.uint32](len(jdb.donor))
            d_jad = ctx.enqueue_create_buffer[DType.uint32](len(jdb.donor))
            ctx.enqueue_copy(d_jd, jdb.donor.unsafe_ptr())
            ctx.enqueue_copy(d_jda, jdb.acc_of_donor.unsafe_ptr())
            ctx.enqueue_copy(d_ja, jdb.acc.unsafe_ptr())
            ctx.enqueue_copy(d_jad, jdb.donor_of_acc.unsafe_ptr())
            ctx.synchronize()
        t0 = perf_counter_ns()
        var base = 0
        while base < n_reads:
            var cnt = min(batch, n_reads - base)
            ctx.enqueue_function[align_kernel](
                d_genome,
                Int32(ref_.n_bases),
                d_bucket,
                d_ehash,
                d_epos,
                d_reads,
                d_rlen,
                Int32(n_reads),
                Int32(base),
                Int32(n_junc),
                d_jd,
                d_jda,
                d_ja,
                d_jad,
                d_mh,
                d_mp,
                d_ak,
                d_aq,
                d_af,
                d_ap,
                d_ac,
                d_out,
                grid_dim=(cnt + BLOCK - 1) // BLOCK,
                block_dim=BLOCK,
            )
            ctx.synchronize()
            base += cnt
        var t_align = secs(t0)
        print("pass", pass_ + 1, ": aligned", n_reads, "reads in", t_align, "s (", Float64(n_reads) / t_align / 1e6, "M reads/s )")

    ctx.enqueue_copy(h_out.unsafe_ptr(), d_out)
    ctx.synchronize()

    # ---- SAM
    t0 = perf_counter_ns()
    var op_chars = String("MIDNS")
    var oc = op_chars.as_bytes()
    var sam = List[UInt8](capacity=n_reads * 110 + 4096)
    for i in range(len(ref_.names)):
        put_str(sam, "@SQ\tSN:" + ref_.names[i] + "\tLN:" + String(ref_.lengths[i]) + "\n")
    put_str(sam, "@PG\tID:gpu_aligner\tPN:gpu_aligner\n")
    var mapped = 0
    for r in range(n_reads):
        var rec = h_out.unsafe_ptr() + r * OUT_STRIDE
        var ok = Int(rec[O_STATUS]) == 1
        var chrom = 0
        var gpos = 0
        if ok:
            gpos = Int(rec[O_POS])
            chrom = ref_.chrom_of(gpos)
            if gpos + Int(rec[O_SPAN]) > ref_.starts[chrom] + ref_.lengths[chrom]:
                ok = False
        for i in range(name_len[r]):
            sam.append(dp[name_start[r] + i])
        if not ok:
            put_str(sam, "\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n")
            continue
        mapped += 1
        sam.append(9)
        put_int(sam, 16 if Int(rec[O_STRAND]) == 1 else 0)
        sam.append(9)
        put_str(sam, ref_.names[chrom])
        sam.append(9)
        put_int(sam, gpos - ref_.starts[chrom] + 1)
        sam.append(9)
        put_int(sam, Int(rec[O_MAPQ]))
        sam.append(9)
        for t in range(Int(rec[O_NOPS])):
            var v = Int(rec[OUT_OPS + t])
            put_int(sam, v >> 3)
            sam.append(oc[v & 7])
        put_str(sam, "\t*\t0\t0\t*\t*\tNM:i:")
        put_int(sam, Int(rec[O_NM]))
        put_str(sam, "\tAS:i:")
        put_int(sam, Int(rec[O_CHAIN]))
        sam.append(10)
    var of = open(String(args[3]), "w")
    of.write_bytes(sam)
    print("mapped", mapped, "of", n_reads, "(", Float64(mapped) * 100.0 / Float64(n_reads), "% ); SAM written in", secs(t0), "s")
