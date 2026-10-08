"""Reference loading and minimizer index construction (host side).

Layout of the index, all flat arrays so they upload straight to the GPU:
    packed      4-bit packed genome, chromosomes concatenated
    bucket_off  2^BUCKET_BITS + 1 offsets into the entry arrays
    ent_hash    32-bit minimizer hash, grouped by bucket (hash >> (32 - BUCKET_BITS))
    ent_pos     k-mer start | strand << 31
"""
from max.gpu.host import DeviceContext, HostBuffer
from std.time import perf_counter_ns

from nt4 import PU8, PU32, nt4_from_ascii, nt4_set, nt4_bytes
from minimizer import scan_minimizers, MM_K, MM_W

comptime BUCKET_BITS = 24
comptime N_BUCKETS = 1 << BUCKET_BITS


struct Reference(Movable):
    var names: List[String]
    var starts: List[Int]
    var lengths: List[Int]
    var n_bases: Int
    var packed: HostBuffer[DType.uint8]

    def __init__(out self, var names: List[String], var starts: List[Int], var lengths: List[Int], n_bases: Int, var packed: HostBuffer[DType.uint8]):
        self.names = names^
        self.starts = starts^
        self.lengths = lengths^
        self.n_bases = n_bases
        self.packed = packed^

    def chrom_of(self, pos: Int) -> Int:
        """Index of the chromosome containing global position pos."""
        var lo = 0
        var hi = len(self.starts) - 1
        while lo < hi:
            var mid = (lo + hi + 1) >> 1
            if self.starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo


struct MinimizerIndex(Movable):
    var n_entries: Int
    var bucket_off: HostBuffer[DType.uint32]
    var ent_hash: HostBuffer[DType.uint32]
    var ent_pos: HostBuffer[DType.uint32]

    def __init__(out self, n_entries: Int, var bucket_off: HostBuffer[DType.uint32], var ent_hash: HostBuffer[DType.uint32], var ent_pos: HostBuffer[DType.uint32]):
        self.n_entries = n_entries
        self.bucket_off = bucket_off^
        self.ent_hash = ent_hash^
        self.ent_pos = ent_pos^


def load_fasta(ctx: DeviceContext, path: String) raises -> Reference:
    """Parse a FASTA file straight into 4-bit packed form."""
    var f = open(path, "r")
    var data = f.read_bytes()
    var packed = ctx.enqueue_create_host_buffer[DType.uint8](nt4_bytes(len(data)) + 1)
    ctx.synchronize()
    var p = packed.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var names = List[String]()
    var starts = List[Int]()
    var lengths = List[Int]()
    var n = 0
    var i = 0
    var size = len(data)
    while i < size:
        if data[i] == 62:  # '>'
            var j = i + 1
            while j < size and data[j] != 10:
                j += 1
            var k = i + 1
            while k < j and data[k] != 32 and data[k] != 9 and data[k] != 13:
                k += 1
            var name = String()
            for t in range(i + 1, k):
                name += chr(Int(data[t]))
            if len(starts) > 0:
                lengths.append(n - starts[len(starts) - 1])
            names.append(name)
            starts.append(n)
            i = j + 1
        else:
            while i < size and data[i] != 10:
                var c = data[i]
                if c != 13:
                    nt4_set(p, n, nt4_from_ascii(c))
                    n += 1
                i += 1
            i += 1
    lengths.append(n - starts[len(starts) - 1])
    return Reference(names^, starts^, lengths^, n, packed^)


def build_index(ctx: DeviceContext, ref_: Reference) raises -> MinimizerIndex:
    var t0 = perf_counter_ns()
    var cap = ref_.n_bases // 3 + 1024
    var tmp_hash = ctx.enqueue_create_host_buffer[DType.uint32](cap)
    var tmp_pos = ctx.enqueue_create_host_buffer[DType.uint32](cap)
    var bucket_off = ctx.enqueue_create_host_buffer[DType.uint32](N_BUCKETS + 1)
    ctx.synchronize()
    var th = tmp_hash.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var tp = tmp_pos.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var bo = bucket_off.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for b in range(N_BUCKETS + 1):
        bo[b] = 0

    var total = 0
    var packed = ref_.packed.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for c in range(len(ref_.starts)):
        var s = ref_.starts[c]
        var e = s + ref_.lengths[c]
        var got = scan_minimizers(packed, s, e, th + total, tp + total, cap - total)
        total += got
    print("  minimizers:", total, "(", Float64(ref_.n_bases) / Float64(total), "bases per minimizer )")

    # Counting sort by bucket.
    for i in range(total):
        bo[Int(th[i] >> UInt32(32 - BUCKET_BITS)) + 1] += 1
    for b in range(N_BUCKETS):
        bo[b + 1] += bo[b]
    var ent_hash = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var ent_pos = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var cursor = ctx.enqueue_create_host_buffer[DType.uint32](N_BUCKETS)
    ctx.synchronize()
    var eh = ent_hash.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ep = ent_pos.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cu = cursor.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for b in range(N_BUCKETS):
        cu[b] = bo[b]
    for i in range(total):
        var b = Int(th[i] >> UInt32(32 - BUCKET_BITS))
        var d = Int(cu[b])
        eh[d] = th[i]
        ep[d] = tp[i]
        cu[b] += 1
    print("  index built in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    return MinimizerIndex(total, bucket_off^, ent_hash^, ent_pos^)
