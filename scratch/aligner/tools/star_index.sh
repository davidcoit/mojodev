#!/bin/bash
# Build STAR indexes for the C. elegans genome: annotated (with GTF junctions) and de novo.
# Usage: tools/star_index.sh [OUT_ROOT]   (run from scratch/aligner)
set -e
STAR=$(pwd)/data/tools/star/STAR_2.7.11b/Linux_x86_64_static/STAR
OUT=${1:-/root/aligner-data/star}
FA=$(pwd)/data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
GTF=$(pwd)/data/raw/Caenorhabditis_elegans.WBcel235.113.gtf
mkdir -p "$OUT/idx_annot" "$OUT/idx_denovo" "$OUT/logs"
# --genomeSAindexNbases = min(14, log2(GenomeLength)/2 - 1) = 12 for 100 Mbp
for MODE in annot denovo; do
    START=$(date +%s%N)
    EXTRA=""
    [ "$MODE" = "annot" ] && EXTRA="--sjdbGTFfile $GTF --sjdbOverhang 100"
    (cd "$OUT/logs" && "$STAR" --runMode genomeGenerate --runThreadN 24 --genomeDir "$OUT/idx_$MODE" \
        --genomeFastaFiles "$FA" --genomeSAindexNbases 12 $EXTRA --outFileNamePrefix "$OUT/logs/index_${MODE}_" > /dev/null)
    END=$(date +%s%N)
    echo "STAR index ($MODE): $(( (END - START) / 1000000 )) ms, $(du -sh "$OUT/idx_$MODE" | cut -f1)"
done
