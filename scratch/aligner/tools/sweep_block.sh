#!/bin/bash
# Build main.mojo with several thread-block sizes and time the alignment passes.
# Usage: tools/sweep_block.sh READS.fastq   (run from scratch/aligner)
set -e
MOJO=${MOJO:-/root/venvs/aligner/bin/mojo}
REF=data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
READS=${1:-data/sim/toy_5m.fastq}
cp src/main.mojo "${TMPDIR:-/tmp}/main.mojo.bak"
for B in 64 128 256 512; do
    sed -i "s/^comptime BLOCK = .*/comptime BLOCK = $B/" src/main.mojo
    "$MOJO" build --disable-warnings -I src src/main.mojo -o "build_blk$B"
    echo "== BLOCK=$B"
    CUDA_VISIBLE_DEVICES=0 "./build_blk$B" "$REF" "$READS" results/sweep.sam | grep -E "^pass [12] :"
done
cp "${TMPDIR:-/tmp}/main.mojo.bak" src/main.mojo
