"""GPU splice-aware aligner (single-end or paired-end).

Usage:
    main REF.fa OUT.sam READS_1.fastq [READS_2.fastq] [options]

Options:
    --chunk N         templates per GPU round trip (default 500000)
    --jn-sample N     templates used for junction discovery before the main pass
                      (default 2000000; 0 disables the junction table)
    --min-support N   reads needed to keep a discovered junction (default 1)
    --max-frag N      largest genomic span of a proper pair (default 60000)

Phase A aligns the first --jn-sample templates de novo and tabulates the splice
junctions they support.  Phase B streams the whole input in chunks, aligning with
that junction table so reads with very short overhangs can be placed, then resolves
pairs and writes SAM.  Mates are stored in adjacent slots (2t, 2t + 1).
"""
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from nt4 import PU8, nt4_bytes
from index import Reference, MinimizerIndex, load_fasta, build_index
from fastq import FastqReader, read_templates, build_lut, NAME_STRIDE
from junctions import JunctionCounter, JunctionDB
from samout import Stats, write_template, put_str
from kernels import (
    align_kernel,
    RSTRIDE,
    MAXL,
    MAXM,
    MAXA,
    REC_STRIDE,
)

comptime BLOCK = 128


def secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


struct GpuAligner(Movable):
    """Device-resident genome, index, junction table and per-chunk buffers."""
    var n_bases: Int
    var batch: Int
    var n_junc: Int
    var kernel_secs: Float64
    var d_genome: DeviceBuffer[DType.uint8]
    var d_bucket: DeviceBuffer[DType.uint32]
    var d_ehash: DeviceBuffer[DType.uint32]
    var d_epos: DeviceBuffer[DType.uint32]
    var d_reads: DeviceBuffer[DType.uint8]
    var d_rlen: DeviceBuffer[DType.int32]
    var d_out: DeviceBuffer[DType.int32]
    var d_mh: DeviceBuffer[DType.uint32]
    var d_mp: DeviceBuffer[DType.uint32]
    var d_ak: DeviceBuffer[DType.uint32]
    var d_aq: DeviceBuffer[DType.int32]
    var d_af: DeviceBuffer[DType.int32]
    var d_ap: DeviceBuffer[DType.int32]
    var d_ac: DeviceBuffer[DType.int32]
    var d_jd: DeviceBuffer[DType.uint32]
    var d_jda: DeviceBuffer[DType.uint32]
    var d_ja: DeviceBuffer[DType.uint32]
    var d_jad: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, ref_: Reference, idx: MinimizerIndex, max_slots: Int, batch: Int) raises:
        self.n_bases = ref_.n_bases
        self.batch = min(batch, max_slots)
        self.n_junc = 0
        self.kernel_secs = 0.0
        self.d_genome = ctx.enqueue_create_buffer[DType.uint8](nt4_bytes(ref_.n_bases) + 1)
        self.d_bucket = ctx.enqueue_create_buffer[DType.uint32](len(idx.bucket_off))
        self.d_ehash = ctx.enqueue_create_buffer[DType.uint32](idx.n_entries)
        self.d_epos = ctx.enqueue_create_buffer[DType.uint32](idx.n_entries)
        ctx.enqueue_copy(self.d_genome, ref_.packed.unsafe_ptr())
        ctx.enqueue_copy(self.d_bucket, idx.bucket_off.unsafe_ptr())
        ctx.enqueue_copy(self.d_ehash, idx.ent_hash.unsafe_ptr())
        ctx.enqueue_copy(self.d_epos, idx.ent_pos.unsafe_ptr())
        self.d_reads = ctx.enqueue_create_buffer[DType.uint8](max_slots * RSTRIDE)
        self.d_rlen = ctx.enqueue_create_buffer[DType.int32](max_slots)
        self.d_out = ctx.enqueue_create_buffer[DType.int32](max_slots * REC_STRIDE)
        self.d_mh = ctx.enqueue_create_buffer[DType.uint32](self.batch * MAXM)
        self.d_mp = ctx.enqueue_create_buffer[DType.uint32](self.batch * MAXM)
        self.d_ak = ctx.enqueue_create_buffer[DType.uint32](self.batch * MAXA)
        self.d_aq = ctx.enqueue_create_buffer[DType.int32](self.batch * MAXA)
        self.d_af = ctx.enqueue_create_buffer[DType.int32](self.batch * MAXA)
        self.d_ap = ctx.enqueue_create_buffer[DType.int32](self.batch * MAXA)
        self.d_ac = ctx.enqueue_create_buffer[DType.int32](self.batch * MAXA)
        self.d_jd = ctx.enqueue_create_buffer[DType.uint32](1)
        self.d_jda = ctx.enqueue_create_buffer[DType.uint32](1)
        self.d_ja = ctx.enqueue_create_buffer[DType.uint32](1)
        self.d_jad = ctx.enqueue_create_buffer[DType.uint32](1)
        ctx.synchronize()

    def set_junctions(mut self, ctx: DeviceContext, db: JunctionDB) raises:
        self.n_junc = db.n
        self.d_jd = ctx.enqueue_create_buffer[DType.uint32](len(db.donor))
        self.d_jda = ctx.enqueue_create_buffer[DType.uint32](len(db.donor))
        self.d_ja = ctx.enqueue_create_buffer[DType.uint32](len(db.donor))
        self.d_jad = ctx.enqueue_create_buffer[DType.uint32](len(db.donor))
        ctx.enqueue_copy(self.d_jd, db.donor.unsafe_ptr())
        ctx.enqueue_copy(self.d_jda, db.acc_of_donor.unsafe_ptr())
        ctx.enqueue_copy(self.d_ja, db.acc.unsafe_ptr())
        ctx.enqueue_copy(self.d_jad, db.donor_of_acc.unsafe_ptr())
        ctx.synchronize()

    def run(
        mut self,
        ctx: DeviceContext,
        h_reads: List[UInt8],
        h_rlen: List[Int32],
        mut h_out: List[Int32],
        n_slots: Int,
    ) raises:
        """Align n_slots reads already packed in h_reads / h_rlen; records land in h_out."""
        ctx.enqueue_copy(self.d_reads, h_reads.unsafe_ptr())
        ctx.enqueue_copy(self.d_rlen, h_rlen.unsafe_ptr())
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var base = 0
        while base < n_slots:
            var cnt = min(self.batch, n_slots - base)
            ctx.enqueue_function[align_kernel](
                self.d_genome,
                Int32(self.n_bases),
                self.d_bucket,
                self.d_ehash,
                self.d_epos,
                self.d_reads,
                self.d_rlen,
                Int32(n_slots),
                Int32(base),
                Int32(cnt),
                Int32(self.n_junc),
                self.d_jd,
                self.d_jda,
                self.d_ja,
                self.d_jad,
                self.d_mh,
                self.d_mp,
                self.d_ak,
                self.d_aq,
                self.d_af,
                self.d_ap,
                self.d_ac,
                self.d_out,
                grid_dim=(cnt + BLOCK - 1) // BLOCK,
                block_dim=BLOCK,
            )
            ctx.synchronize()
            base += cnt
        self.kernel_secs += secs(t0)
        ctx.enqueue_copy(h_out.unsafe_ptr(), self.d_out)
        ctx.synchronize()


def open_readers(path1: String, path2: String, paired: Bool) raises -> List[FastqReader]:
    var rs = List[FastqReader]()
    rs.append(FastqReader(path1))
    if paired:
        rs.append(FastqReader(path2))
    return rs^


def main() raises:
    var args = argv()
    var pos_args = List[String]()
    var chunk = 500000
    var jn_sample = 2000000
    var min_support = 1
    var max_frag = 60000
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--chunk":
            chunk = Int(String(args[i + 1]))
            i += 2
        elif a == "--jn-sample":
            jn_sample = Int(String(args[i + 1]))
            i += 2
        elif a == "--min-support":
            min_support = Int(String(args[i + 1]))
            i += 2
        elif a == "--max-frag":
            max_frag = Int(String(args[i + 1]))
            i += 2
        else:
            pos_args.append(a)
            i += 1
    if len(pos_args) < 3 or len(pos_args) > 4:
        print("usage: main REF.fa OUT.sam READS_1.fastq [READS_2.fastq] [--chunk N] [--jn-sample N] [--min-support N] [--max-frag N]")
        return
    var paired = len(pos_args) == 4
    var path2 = pos_args[3] if paired else String("")
    var stride = 2 if paired else 1

    var t_all = perf_counter_ns()
    var ctx = DeviceContext()
    print("device:", ctx.name(), "|", "paired-end" if paired else "single-end")

    var t0 = perf_counter_ns()
    var ref_ = load_fasta(pos_args[0])
    print("reference:", len(ref_.names), "sequences,", ref_.n_bases, "bases (", secs(t0), "s )")
    var idx = build_index(ref_)

    var max_slots = chunk * stride
    var ga = GpuAligner(ctx, ref_, idx, max_slots, 200000)
    var h_reads = List[UInt8](length=max_slots * RSTRIDE, fill=0)
    var h_rlen = List[Int32](length=max_slots, fill=0)
    var h_out = List[Int32](length=max_slots * REC_STRIDE, fill=0)
    var lut = build_lut()
    var names = List[UInt8](length=chunk * NAME_STRIDE, fill=0)
    var name_off = List[Int](length=chunk, fill=0)
    var name_len = List[Int](length=chunk, fill=0)
    for t in range(chunk):
        name_off[t] = t * NAME_STRIDE

    # ---- phase A: discover junctions from the first jn_sample templates (de novo pass)
    if jn_sample > 0:
        t0 = perf_counter_ns()
        var readers_a = open_readers(pos_args[2], path2, paired)
        var counter = JunctionCounter()
        var remaining = jn_sample
        var seen = 0
        while remaining > 0:
            var n = read_templates(readers_a, paired, min(chunk, remaining), lut, h_reads, h_rlen, names, name_len)
            if n == 0:
                break
            ga.run(ctx, h_reads, h_rlen, h_out, n * stride)
            counter.add(h_out, n * stride, 12)
            remaining -= n
            seen += n
        var db = counter.build(min_support, ref_.packed.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), ref_.n_bases)
        ga.set_junctions(ctx, db)
        print("phase A:", seen, "templates ->", db.n, "junctions with >=", min_support, "supporting reads (", secs(t0), "s )")

    # ---- phase B: stream everything, align with the junction table, resolve pairs, write SAM
    var readers = open_readers(pos_args[2], path2, paired)
    var of = open(pos_args[1], "w")
    var sam = List[UInt8](capacity=chunk * 220 + 4096)
    for c in range(len(ref_.names)):
        put_str(sam, "@SQ\tSN:" + ref_.names[c] + "\tLN:" + String(ref_.lengths[c]) + "\n")
    put_str(sam, "@PG\tID:gpu_aligner\tPN:gpu_aligner\n")
    of.write_bytes(sam)
    sam.clear()
    var stats = Stats()
    var t_parse = 0.0
    var t_pair = 0.0
    ga.kernel_secs = 0.0
    var total_t = 0
    while True:
        t0 = perf_counter_ns()
        var n = read_templates(readers, paired, chunk, lut, h_reads, h_rlen, names, name_len)
        t_parse += secs(t0)
        if n == 0:
            break
        ga.run(ctx, h_reads, h_rlen, h_out, n * stride)
        t0 = perf_counter_ns()
        for t in range(n):
            write_template(sam, h_out, t, paired, max_frag, ref_, names, name_off[t], name_len[t], stats)
        of.write_bytes(sam)
        sam.clear()
        t_pair += secs(t0)
        total_t += n
    of.close()

    print("phase B: parse", t_parse, "s | GPU kernels", ga.kernel_secs, "s | pairing + SAM", t_pair, "s")
    var n_reads = total_t * stride
    print("GPU alignment throughput:", Float64(n_reads) / ga.kernel_secs / 1e6, "M reads/s (", total_t, "templates,", n_reads, "reads )")
    var tt = Float64(max(stats.templates, 1))
    if paired:
        print("mate 1 mapped:", 100.0 * Float64(stats.mate1_mapped) / tt, "% | mate 2 mapped:", 100.0 * Float64(stats.mate2_mapped) / tt, "%")
        print("both mapped:", 100.0 * Float64(stats.both_mapped) / tt, "% | proper pairs:", 100.0 * Float64(stats.proper) / tt, "% | one mate only:", 100.0 * Float64(stats.one_mapped) / tt, "% | neither:", 100.0 * Float64(stats.unmapped) / tt, "%")
    else:
        print("mapped:", 100.0 * Float64(stats.mate1_mapped) / tt, "%")
    print("reads with a splice junction:", 100.0 * Float64(stats.spliced_mates) / Float64(max(stats.reads_total, 1)), "%")
    print("total wall time:", secs(t_all), "s")
