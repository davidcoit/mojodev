#!/bin/bash
# minimap2 splice:sr at full thread count on a large read set (and an index-only timing).
# Usage: tools/bench_minimap2_big.sh READS.fastq TRUTH.tsv TAG   (run from scratch/aligner)
set -e
MM=data/tools/minimap2/minimap2-2.28_x64-linux/minimap2
REF=data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
READS=$1
TRUTH=$2
TAG=$3
# index only: an empty read set costs just the index build
: > results/empty.fastq
START=$(date +%s%N)
"$MM" -ax splice:sr --secondary=no -t 24 "$REF" results/empty.fastq -o /dev/null 2>/dev/null
END=$(date +%s%N)
echo "minimap2 index build + startup: $(( (END - START) / 1000000 )) ms"
START=$(date +%s%N)
"$MM" -ax splice:sr --secondary=no -t 24 "$REF" "$READS" -o "results/mm2_${TAG}_t24.sam" 2>/dev/null
END=$(date +%s%N)
echo "minimap2 splice:sr -t 24 total: $(( (END - START) / 1000000 )) ms wall"
python3 -I tools/score_sam.py "results/mm2_${TAG}_t24.sam" "$TRUTH"
