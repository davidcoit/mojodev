# gen.mojo -- deterministic test-data generators for the TE estimator.
#
# The PRNG is a plain xorshift64* so that every run is reproducible without
# depending on the stdlib's global RNG state.

from bitte import BitSeq


struct Rng(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed if seed != 0 else 0x9E3779B97F4A7C15

    def next(mut self) -> UInt64:
        var x = self.s
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.s = x
        return x * 2685821657736338717

    def unit(mut self) -> Float64:
        """Uniform in [0, 1) from the top 53 bits."""
        return Float64(self.next() >> 11) * (1.0 / 9007199254740992.0)

    def bit(mut self, p: Float64) -> Int:
        return 1 if self.unit() < p else 0


def iid(n: Int, p: Float64, mut rng: Rng) -> List[Int]:
    """n independent Bernoulli(p) bits."""
    var out = List[Int](length=n, fill=0)
    for i in range(n):
        out[i] = rng.bit(p)
    return out^


def iid_packed(n: Int, mut rng: Rng) -> BitSeq:
    """Fair coin flips generated straight into packed words, 64 bits per call.

    Nothing in the estimator needs an unpacked copy, so for a pure bitwise
    pipeline this is the natural source: no per-sample branch anywhere.
    """
    var s = BitSeq(n)
    var full = (n + 63) // 64
    for w in range(full):
        s.w[w] = rng.next()
    var rem = n & 63
    if rem != 0:
        s.w[full - 1] &= (UInt64(1) << UInt64(rem)) - 1
    return s^


def channel(
    x: List[Int], u: Int, eps: Float64, mut rng: Rng
) -> List[Int]:
    """y[t] = x[t-u] XOR Bernoulli(eps); y[t] = fresh coin while t < u.

    Matches the estimator's convention: with source lag u and l = 1 the
    relevant source bit for y[n+1] is x[n-u+1] = x[(n+1)-u].
    """
    var n = len(x)
    var y = List[Int](length=n, fill=0)
    for t in range(n):
        if t < u:
            y[t] = rng.bit(0.5)
        else:
            y[t] = x[t - u] ^ rng.bit(eps)
    return y^


def xor_channel2(x: List[Int], mut rng: Rng) -> List[Int]:
    """y[t] = x[t-1] XOR x[t-2]: invisible to l = 1, fully explained by l = 2.

    Either source bit alone leaves y' a fair coin, so this separates a real
    source-history length from a spurious one.
    """
    var n = len(x)
    var y = List[Int](length=n, fill=0)
    for t in range(n):
        if t < 2:
            y[t] = rng.bit(0.5)
        else:
            y[t] = x[t - 1] ^ x[t - 2]
    return y^


def autoregressive(n: Int, flip: Float64, mut rng: Rng) -> List[Int]:
    """A target with strong memory of its own past and no source input at all.

    y[t] = y[t-1] with probability 1 - flip.  Any estimator that confuses
    'predictable' with 'driven' will report a large TE here.
    """
    var y = List[Int](length=n, fill=0)
    y[0] = rng.bit(0.5)
    for t in range(1, n):
        y[t] = y[t - 1] ^ rng.bit(flip)
    return y^
