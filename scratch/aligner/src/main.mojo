"""GPU splice-aware aligner.

Usage: mojo run -I src src/main.mojo REF.fa READS.fastq OUT.sam [batch_reads]
"""
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from nt4 import nt4_from_ascii, nt4_set, nt4_bytes
from index import load_fasta, build_index
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


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("usage: main REF.fa READS.fastq OUT.sam [batch_reads]")
        return
    var batch = 200000
    if len(args) > 4:
        batch = Int(String(args[4]))

    var ctx = DeviceContext()
    print("device:", ctx.name())

    # ---- reference + index
    var t0 = perf_counter_ns()
    var ref_ = load_fasta(ctx, String(args[1]))
    print("reference:", len(ref_.names), "sequences,", ref_.n_bases, "bases (", secs(t0), "s )")
    var idx = build_index(ctx, ref_)

    var d_genome = ctx.enqueue_create_buffer[DType.uint8](nt4_bytes(ref_.n_bases) + 1)
    var d_bucket = ctx.enqueue_create_buffer[DType.uint32](len(idx.bucket_off))
    var d_ehash = ctx.enqueue_create_buffer[DType.uint32](idx.n_entries)
    var d_epos = ctx.enqueue_create_buffer[DType.uint32](idx.n_entries)
    ctx.enqueue_copy(d_genome, ref_.packed)
    ctx.enqueue_copy(d_bucket, idx.bucket_off)
    ctx.enqueue_copy(d_ehash, idx.ent_hash)
    ctx.enqueue_copy(d_epos, idx.ent_pos)
    ctx.synchronize()
    var index_mb = Float64(len(idx.bucket_off) * 4 + idx.n_entries * 8 + nt4_bytes(ref_.n_bases)) / 1e6
    print("device index:", index_mb, "MB")

    # ---- reads -> 4-bit packed slots
    t0 = perf_counter_ns()
    var fq = open(String(args[2]), "r")
    var data = fq.read_bytes()
    var names = List[String]()
    var n_reads = 0
    var count = 0
    for i in range(len(data)):
        if data[i] == 10:
            count += 1
    n_reads = count // 4
    var h_reads = ctx.enqueue_create_host_buffer[DType.uint8](n_reads * RSTRIDE)
    var h_rlen = ctx.enqueue_create_host_buffer[DType.int32](n_reads)
    ctx.synchronize()
    var rp = h_reads.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for i in range(n_reads * RSTRIDE):
        rp[i] = 0
    var pos = 0
    var size = len(data)
    for r in range(n_reads):
        # header
        var nm = String()
        pos += 1
        while data[pos] != 10 and data[pos] != 32:
            nm += chr(Int(data[pos]))
            pos += 1
        while data[pos] != 10:
            pos += 1
        pos += 1
        names.append(nm)
        # sequence
        var L = 0
        var slot = rp + r * RSTRIDE
        while data[pos] != 10:
            if L < MAXL:
                nt4_set(slot, L, nt4_from_ascii(data[pos]))
            L += 1
            pos += 1
        pos += 1
        # '+' line and qualities
        while data[pos] != 10:
            pos += 1
        pos += 1
        while data[pos] != 10:
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
    ctx.enqueue_copy(d_reads, h_reads)
    ctx.enqueue_copy(d_rlen, h_rlen)
    ctx.synchronize()

    # ---- align in batches
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
    print("aligned", n_reads, "reads in", t_align, "s (", Float64(n_reads) / t_align / 1e6, "M reads/s )")

    var h_out = ctx.enqueue_create_host_buffer[DType.int32](n_reads * OUT_STRIDE)
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # ---- SAM
    t0 = perf_counter_ns()
    var op_chars = String("MIDNS")
    var sam = String()
    sam.reserve(n_reads * 100)
    for i in range(len(ref_.names)):
        sam += "@SQ\tSN:" + ref_.names[i] + "\tLN:" + String(ref_.lengths[i]) + "\n"
    sam += "@PG\tID:gpu_aligner\tPN:gpu_aligner\n"
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
        if not ok:
            sam += names[r] + "\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n"
            continue
        mapped += 1
        var flag = 0
        if Int(rec[O_STRAND]) == 1:
            flag = 16
        var cigar = String()
        for t in range(Int(rec[O_NOPS])):
            var v = Int(rec[OUT_OPS + t])
            cigar += String(v >> 3)
            cigar += chr(Int(op_chars.as_bytes()[v & 7]))
        sam += names[r] + "\t" + String(flag) + "\t" + ref_.names[chrom] + "\t"
        sam += String(gpos - ref_.starts[chrom] + 1) + "\t" + String(Int(rec[O_MAPQ])) + "\t"
        sam += cigar + "\t*\t0\t0\t*\t*\tNM:i:" + String(Int(rec[O_NM])) + "\tAS:i:" + String(Int(rec[O_CHAIN])) + "\n"
    var of = open(String(args[3]), "w")
    of.write(sam)
    print("mapped", mapped, "of", n_reads, "(", Float64(mapped) * 100.0 / Float64(n_reads), "% ); SAM written in", secs(t0), "s")
