from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from index import load_fasta, build_index
from nt4 import nt4_get, nt4_to_ascii


def main() raises:
    var ctx = DeviceContext()
    var t0 = perf_counter_ns()
    var ref_ = load_fasta("data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa")
    print("loaded", len(ref_.names), "sequences,", ref_.n_bases, "bases in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    for i in range(len(ref_.names)):
        print(" ", ref_.names[i], ref_.starts[i], ref_.lengths[i])
    var p = ref_.packed.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var s = String()
    for i in range(60):
        s += chr(Int(nt4_to_ascii(nt4_get(p, i))))
    print(s)
    var idx = build_index(ref_)
    print("entries:", idx.n_entries)
