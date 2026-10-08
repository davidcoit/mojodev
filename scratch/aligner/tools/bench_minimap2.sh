#!/bin/bash
# CPU baseline: minimap2 in short-read splice mode.
# Usage: tools/bench_minimap2.sh READS.fastq TRUTH.tsv TAG   (run from scratch/aligner)
set -e
MM=data/tools/minimap2/minimap2-2.28_x64-linux/minimap2
REF=data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
READS=$1
TRUTH=$2
TAG=$3
for T in 1 24; do
    START=$(date +%s%N)
    "$MM" -ax splice:sr --secondary=no -t "$T" "$REF" "$READS" -o "results/mm2_${TAG}_t$T.sam" 2>/dev/null
    END=$(date +%s%N)
    echo "minimap2 splice:sr -t $T: $(( (END - START) / 1000000 )) ms wall (includes index build)"
done
python3 -I tools/score_sam.py "results/mm2_${TAG}_t24.sam" "$TRUTH"
