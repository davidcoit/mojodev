#!/bin/bash
# Repeated alignment-only timings (3x by default) for the median/range the comparison needs.
# Usage: tools/repeat_timing.sh NAME R1 R2 OUTROOT [REPS] [ARMS]   (run from scratch/aligner)
set -e
NAME=$1; R1=$2; R2=$3; ROOT=$4; REPS=${5:-3}; ARMS=${6:-"gpu star_annot star_annot40k star_denovo2p"}
for i in $(seq 1 "$REPS"); do
    NOCOUNT=1 CLEAN=1 bash tools/run_compare.sh "$NAME" "$R1" "$R2" "$ROOT/rep$i" "$ARMS"
done
