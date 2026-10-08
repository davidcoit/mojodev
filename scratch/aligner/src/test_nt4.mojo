"""Unit tests for the 4-bit nucleotide type and the canonical minimizer scanner.

Run: mojo run --disable-warnings -I src src/test_nt4.mojo
"""
from nt4 import (
    PU8,
    PU32,
    nt4_from_ascii,
    nt4_to_ascii,
    nt4_complement,
    nt4_get,
    nt4_set,
    nt4_bytes,
    NT_N,
)
from minimizer import scan_minimizers, MM_K, MM_W


def check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("FAILED: " + msg)


def pack(seq: String) -> List[UInt8]:
    var b = seq.as_bytes()
    var out = List[UInt8](length=nt4_bytes(len(b)) + 1, fill=0)
    var p = out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for i in range(len(b)):
        nt4_set(p, i, nt4_from_ascii(b[i]))
    return out^


def revcomp(seq: String) -> String:
    var b = seq.as_bytes()
    var out = String()
    for i in range(len(b) - 1, -1, -1):
        var c = nt4_to_ascii(nt4_complement(nt4_from_ascii(b[i])))
        out += chr(Int(c))
    return out


def lcg_seq(n: Int, seed: Int) -> String:
    var s = String()
    var x = UInt64(seed)
    for _ in range(n):
        x = x * 6364136223846793005 + 1442695040888963407
        s += chr(Int(nt4_to_ascii(UInt8((x >> 33) & 3))))
    return s


def minimizer_hashes(seq: String) -> List[UInt32]:
    var packed = pack(seq)
    var n = len(seq.as_bytes())
    var hs = List[UInt32](length=n, fill=0)
    var ps = List[UInt32](length=n, fill=0)
    var cnt = scan_minimizers(
        packed.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](),
        0,
        n,
        hs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        ps.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        n,
    )
    var out = List[UInt32]()
    for i in range(cnt):
        out.append(hs[i])
    return out^


def contains(xs: List[UInt32], v: UInt32) -> Bool:
    for i in range(len(xs)):
        if xs[i] == v:
            return True
    return False


def main() raises:
    # --- round trip through the packed representation (odd length, mixed case, N)
    var text = String("ACGTNacgtnACGTACGTA")
    var packed = pack(text)
    var p = packed.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var tb = text.as_bytes()
    for i in range(len(tb)):
        var want = nt4_to_ascii(nt4_from_ascii(tb[i]))
        check(nt4_to_ascii(nt4_get(p, i)) == want, "round trip at " + String(i))
    check(nt4_get(p, 4) == NT_N and nt4_get(p, 9) == NT_N, "N survives packing")
    check(len(packed) == nt4_bytes(len(tb)) + 1, "packed size")
    print("nt4 packing: ok (", len(tb), "bases in", nt4_bytes(len(tb)), "bytes )")

    # --- complement
    check(nt4_complement(0) == 3 and nt4_complement(1) == 2, "complement A/C")
    check(nt4_complement(NT_N) == NT_N, "complement N")
    print("complement: ok")

    # --- overwriting one nibble must not disturb its neighbour
    var before1 = nt4_get(p, 1)
    var before3 = nt4_get(p, 3)
    nt4_set(p, 2, 3)
    check(nt4_get(p, 2) == 3, "set took effect")
    check(nt4_get(p, 1) == before1 and nt4_get(p, 3) == before3, "neighbours intact after set")
    nt4_set(p, 3, 1)
    check(nt4_get(p, 2) == 3 and nt4_get(p, 3) == 1, "adjacent nibbles independent")
    print("nibble isolation: ok")

    # --- canonical minimizers: a sequence and its reverse complement agree
    var seq = lcg_seq(2000, 42)
    var fwd = minimizer_hashes(seq)
    var rev = minimizer_hashes(revcomp(seq))
    check(len(fwd) > 100, "enough minimizers")
    var shared = 0
    for i in range(len(fwd)):
        if contains(rev, fwd[i]):
            shared += 1
    # windows differ only at the two ends, so nearly all hashes must be shared
    check(shared * 100 >= len(fwd) * 95, "fwd/revcomp minimizer overlap " + String(shared) + "/" + String(len(fwd)))
    print("canonical minimizers: ok (", shared, "of", len(fwd), "shared with reverse complement )")

    # --- density close to 2/(w+1)
    var density = Float64(len(fwd)) / 2000.0
    var expect = 2.0 / Float64(MM_W + 1)
    check(density > 0.8 * expect and density < 1.2 * expect, "minimizer density")
    print("density:", density, "(expected ~", expect, ")")

    # --- N breaks k-mers: no minimizer may span an N
    var with_n = String()
    var sb = seq.as_bytes()
    for i in range(len(sb)):
        if i == 500:
            with_n += "N"
        else:
            with_n += chr(Int(sb[i]))
    var h_n = minimizer_hashes(with_n)
    check(len(h_n) < len(fwd) + 5 and len(h_n) > 0, "N handling")
    print("N handling: ok")
    print("ALL TESTS PASSED")
