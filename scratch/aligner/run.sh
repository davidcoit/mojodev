#!/bin/bash
# Build, simulate (if needed), align, and score.  Run from scratch/aligner.
#   ./run.sh [N_READS]          default 200000
set -e
N=${1:-200000}
MOJO=${MOJO:-/root/venvs/aligner/bin/mojo}
REF=data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
GTF=data/raw/Caenorhabditis_elegans.WBcel235.113.gtf
PREFIX=data/sim/toy_$N

# This container exports CUDA_VISIBLE_DEVICES=all, which is not a valid value and
# makes cuInit report "no device".  Pin it to the first GPU.
export CUDA_VISIBLE_DEVICES=0

if [ ! -f "$PREFIX.fastq" ]; then
    python3 -I tools/simulate_reads.py "$REF" "$GTF" "$PREFIX" "$N" 100 1
fi
"$MOJO" build --disable-warnings -I src src/main.mojo -o build_main
./build_main "$REF" "$PREFIX.fastq" "results/toy_$N.sam"
python3 -I tools/score_sam.py "results/toy_$N.sam" "$PREFIX.truth.tsv"
