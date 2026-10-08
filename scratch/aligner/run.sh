#!/bin/bash
# Build, simulate (if needed), align, and score.  Run from scratch/aligner.
#   ./run.sh [N] [se|pe]        default: 200000 se
#   se: N simulated single-end reads;  pe: N simulated read pairs
set -e
N=${1:-200000}
MODE=${2:-se}
MOJO=${MOJO:-/root/venvs/aligner/bin/mojo}
REF=data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
GTF=data/raw/Caenorhabditis_elegans.WBcel235.113.gtf

# This container exports CUDA_VISIBLE_DEVICES=all, which is not a valid value and
# makes cuInit report "no device".  Pin it to the first GPU.
export CUDA_VISIBLE_DEVICES=0

"$MOJO" build --disable-warnings -I src src/main.mojo -o build_main
mkdir -p results data/sim
if [ "$MODE" = "pe" ]; then
    PREFIX=data/sim/pe_$N
    [ -f "${PREFIX}_1.fastq" ] || python3 -I tools/simulate_pairs.py "$REF" "$GTF" "$PREFIX" "$N" 100 11
    ./build_main "$REF" "results/pe_$N.sam" "${PREFIX}_1.fastq" "${PREFIX}_2.fastq"
    python3 -I tools/score_sam.py "results/pe_$N.sam" "$PREFIX.truth.tsv"
else
    PREFIX=data/sim/toy_$N
    [ -f "$PREFIX.fastq" ] || python3 -I tools/simulate_reads.py "$REF" "$GTF" "$PREFIX" "$N" 100 1
    ./build_main "$REF" "results/toy_$N.sam" "$PREFIX.fastq"
    python3 -I tools/score_sam.py "results/toy_$N.sam" "$PREFIX.truth.tsv"
fi
