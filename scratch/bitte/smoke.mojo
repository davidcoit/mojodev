from bitte import BitSeq, BitTE, te_naive, f6, slice_into


def main() raises:
    # Hand-checkable case: y[n+1] = x[n], so TE(x->y) with k=l=u=1 should be
    # exactly 1 bit if x is balanced and y gives no hint about its own future.
    var base = [0, 1, 1, 0, 1, 0, 0, 1]
    var xs = List[Int]()
    for _r in range(40):
        for i in range(8):
            xs.append(base[i])
    var ys = List[Int]()
    ys.append(0)
    for i in range(len(xs) - 1):
        ys.append(xs[i])

    var y = BitSeq(ys)
    var x = BitSeq(xs)
    print("n =", y.n, " x[0..4] =", x.get(0), x.get(1), x.get(2), x.get(3))

    # slice sanity: a plane of x at offset 2 must equal x[2], x[3], ...
    var buf = List[UInt64](length=4, fill=0)
    slice_into(x, 2, 200, buf, 0)
    var ok = True
    for i in range(200):
        var got = Int((buf[i >> 6] >> UInt64(i & 63)) & 1)
        if got != x.get(i + 2):
            ok = False
    print("slice_into offset-2 matches:", ok)

    var est = BitTE(y, x, 1, 1, 1)
    est.count()
    print("m =", est.m, " total counted =", est.total_counted())
    print("occupied bins =", est.occupied())
    print("TE bitwise =", f6(est.te_bits()))
    print("TE naive   =", f6(te_naive(ys, xs, 1, 1, 1)))
