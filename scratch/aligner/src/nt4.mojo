"""4-bit nucleotide encoding: A, C, G, T, N packed two bases per byte.

Codes: A=0 C=1 G=2 T=3 N=4 (values 5..15 are reserved for IUPAC codes later).
Low nibble holds the even-indexed base, high nibble the odd-indexed one, so a
sequence of n bases occupies (n + 1) // 2 bytes.  Complement of a real base is
`3 - code`; N stays N.

The free functions below take raw pointers so the very same code runs on the
host and inside GPU kernels.
"""

comptime PU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PU32 = UnsafePointer[UInt32, MutAnyOrigin]
comptime PI32 = UnsafePointer[Int32, MutAnyOrigin]

comptime NT_A: UInt8 = 0
comptime NT_C: UInt8 = 1
comptime NT_G: UInt8 = 2
comptime NT_T: UInt8 = 3
comptime NT_N: UInt8 = 4


@always_inline
def nt4_from_ascii(c: UInt8) -> UInt8:
    """ASCII -> 4-bit code.  Anything that is not ACGT (either case) becomes N."""
    var u = c & 0xDF  # fold to upper case
    if u == 65:
        return NT_A
    if u == 67:
        return NT_C
    if u == 71:
        return NT_G
    if u == 84:
        return NT_T
    return NT_N


@always_inline
def nt4_to_ascii(code: UInt8) -> UInt8:
    if code == NT_A:
        return 65
    if code == NT_C:
        return 67
    if code == NT_G:
        return 71
    if code == NT_T:
        return 84
    return 78


@always_inline
def nt4_complement(code: UInt8) -> UInt8:
    if code > 3:
        return code
    return 3 - code


@always_inline
def nt4_get(packed: PU8, i: Int) -> UInt8:
    """Base i of a packed sequence."""
    return (packed[i >> 1] >> UInt8((i & 1) << 2)) & 0xF


@always_inline
def nt4_set(packed: PU8, i: Int, code: UInt8):
    """Write base i of a packed sequence (host side, single writer per byte)."""
    var b = packed[i >> 1]
    if (i & 1) == 0:
        packed[i >> 1] = (b & 0xF0) | (code & 0xF)
    else:
        packed[i >> 1] = (b & 0x0F) | ((code & 0xF) << 4)


@always_inline
def nt4_bytes(n_bases: Int) -> Int:
    return (n_bases + 1) >> 1
