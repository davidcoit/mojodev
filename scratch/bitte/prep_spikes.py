#!/usr/bin/env python3
"""Rough-and-ready spike detection from calcium fluorescence -> packed bits.

    <venv>/bin/python -I prep_spikes.py <fluorescence.txt> <out.bits> [--z 2.5]

Input is a text matrix, rows = time points, columns = samples (neurons),
comma separated. Output is the bit-packed format `tespikes.mojo` reads, which
is byte-for-byte the layout of `BitSeq.w` in bitte.mojo:

    int64  magic = 20261015
    int64  p          number of signals (columns)
    int64  n          samples per signal (rows, after differencing)
    int64  wps        uint64 words per signal = (n + 63) // 64 + 1
    uint64 words[p * wps]    signal-major; bit i of signal j at
                             words[j*wps + (i >> 6)] bit (i & 63), LSB-first.
                             The final word of each signal is BitSeq's zero
                             guard word, which the funnel-shift slice reads.

WHY THIS DETECTOR
-----------------
A calcium indicator integrates spikes and decays slowly, so the raw trace is
a smoothed, drifting version of the spike train: thresholding it directly
mostly detects "was recently active". The first difference along time removes
the decay and the drift and leaves a sharp positive transient at each spike,
which is what gets thresholded. This is the standard cheap baseline for this
kind of data; it is not a deconvolution and makes no attempt to resolve spike
counts within a frame.

Two details that matter for the entropy estimate:

  * The threshold is per-neuron, from the median and MAD of that neuron's
    own derivative, so a neuron with a noisier trace does not simply look
    more active.
  * By default only the *rising edge* of each supra-threshold run is kept, so
    one transient yields one event rather than two or three adjacent 1s.
    Without this, the target's own history bit becomes strongly predictive of
    its next bit purely through detector persistence, which inflates the
    conditional entropy the TE is measured against.
"""

import argparse
import struct
import sys
import time

import numpy as np

MAGIC = 20261015


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--z", type=float, default=2.5,
                    help="robust z threshold on the derivative (default 2.5)")
    ap.add_argument("--no-edges", action="store_true",
                    help="keep every supra-threshold frame, not just onsets")
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="use only the first N rows (for quick tests)")
    args = ap.parse_args()

    t0 = time.time()
    if args.limit_rows:
        rows = []
        with open(args.infile, "r") as f:
            for i, line in enumerate(f):
                if i >= args.limit_rows:
                    break
                rows.append(line)
        f_raw = np.loadtxt(rows, delimiter=",", dtype=np.float32)
    else:
        f_raw = np.loadtxt(args.infile, delimiter=",", dtype=np.float32)
    t_load = time.time() - t0
    if f_raw.ndim != 2:
        print("expected a 2-D matrix, got shape %r" % (f_raw.shape,))
        return 1
    nt, p = f_raw.shape
    print("loaded %d time points x %d samples in %.2f s"
          % (nt, p, t_load))

    # --- detect ----------------------------------------------------------
    t1 = time.time()
    d = np.diff(f_raw, axis=0)                 # (nt-1, p)
    med = np.median(d, axis=0)
    mad = np.median(np.abs(d - med), axis=0)
    sigma = np.maximum(1.4826 * mad, 1e-9)     # MAD -> sigma for a normal
    z = (d - med) / sigma
    hot = z > args.z

    if not args.no_edges:
        # Rising edges only: a frame counts if it is hot and the one before
        # it was not.  Column-wise, so no leakage between neurons.
        prev = np.vstack([np.zeros((1, p), dtype=bool), hot[:-1]])
        spikes = hot & ~prev
    else:
        spikes = hot
    t_detect = time.time() - t1

    n = spikes.shape[0]
    per = spikes.mean(axis=0)
    total = int(spikes.sum())
    print("detected in %.2f s" % t_detect)
    print("  threshold: robust z > %.2f on the first difference%s"
          % (args.z, "" if args.no_edges else ", rising edges only"))
    print("  %d events over %d frames x %d samples" % (total, n, p))
    print("  density %.5f  (%.3f Hz at 50 Hz frames)"
          % (spikes.mean(), spikes.mean() * 50.0))
    print("  per-sample density: min %.5f  median %.5f  max %.5f"
          % (per.min(), np.median(per), per.max()))
    print("  events per sample: min %d  median %d  max %d"
          % (spikes.sum(0).min(), int(np.median(spikes.sum(0))),
             spikes.sum(0).max()))

    # Population synchrony is worth knowing before reading any TE number:
    # this kind of culture fires in network-wide bursts, which couples every
    # pair whether or not they are wired together.
    pop = spikes.sum(axis=1)
    print("  population: frames with 0 events %.3f, >=10 events %.5f, "
          "max in one frame %d"
          % ((pop == 0).mean(), (pop >= 10).mean(), pop.max()))

    # --- pack ------------------------------------------------------------
    t2 = time.time()
    wps = (n + 63) // 64 + 1                  # + BitSeq's guard word
    out = np.zeros((p, wps), dtype=np.uint64)
    padded = np.zeros(wps * 64, dtype=np.uint8)
    for j in range(p):
        padded[:] = 0
        padded[:n] = spikes[:, j].view(np.uint8)
        # bitorder='little' puts element i at bit (i & 7) of byte (i >> 3);
        # viewing 8 little-endian bytes as a uint64 then puts element i at
        # bit (i & 63) of word (i >> 6), which is exactly BitSeq's layout.
        out[j] = np.packbits(padded, bitorder="little").view(np.uint64)
    t_pack = time.time() - t2

    with open(args.outfile, "wb") as fh:
        fh.write(struct.pack("<4q", MAGIC, p, n, wps))
        fh.write(out.tobytes(order="C"))
    print("packed in %.2f s -> %s (%.1f KB for %d signals)"
          % (t_pack, args.outfile,
             (32 + out.nbytes) / 1024.0, p))
    print("  raw text was %.1f MB; packed is %.0fx smaller"
          % (nt * p * 7 / 1048576.0,
             (nt * p * 7) / float(out.nbytes)))

    # Verify the packing round-trips before anything downstream trusts it.
    bad = 0
    for j in range(0, p, max(1, p // 8)):
        w = out[j]
        idx = np.arange(n)
        got = (w[idx >> 6] >> (idx & 63).astype(np.uint64)) & np.uint64(1)
        if not np.array_equal(got.astype(bool), spikes[:, j]):
            bad += 1
    print("  round-trip check on %d sampled signals: %s"
          % (len(range(0, p, max(1, p // 8))),
             "OK" if bad == 0 else "FAILED (%d)" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
