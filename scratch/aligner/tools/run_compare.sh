#!/bin/bash
# Align one paired-end dataset with every arm, then count each alignment with featureCounts.
# Usage: tools/run_compare.sh NAME R1.fastq R2.fastq OUTDIR [ARMS]   (run from scratch/aligner)
#   ARMS: space-separated subset of: truth gpu star_annot star_annot40k star_denovo2p   (default: all)
#   Env: NOCOUNT=1 skips featureCounts (alignment timing only); CLEAN=1 deletes the SAM afterwards.
#   truth = counts the simulator's perfect-alignment SAM; needs TRUTH_SAM=/path/to/prefix.truth.sam
#
# For every arm: <OUTDIR>/<arm>/{aligned.sam, counts_*.txt, ...} and timings appended to
# <OUTDIR>/timings.jsonl.  featureCounts variants (unstranded library, fragments = read pairs):
#   all    : -O/-M off, exon/gene_id, all genes          (default policy: unique, no overlaps)
#   frac   : -O -M --fraction, all genes                 (share multi-overlap / multi-mappers)
#   pc     : default policy on a protein-coding-only GTF (removes most overlapping-gene ambiguity)
#   core   : default policy + -R CORE for per-read assignments (scored against simulated truth)
set -e
NAME=$1; R1=$2; R2=$3; OUT=$4
ARMS=${5:-"truth gpu star_annot star_annot40k star_denovo2p"}
HERE=$(pwd)
TI="python3 -I $HERE/tools/timeit.py"
STAR=$HERE/data/tools/star/STAR_2.7.11b/Linux_x86_64_static/STAR
FC=$HERE/data/tools/featurecounts-bin/featureCounts
GTF=$HERE/data/raw/Caenorhabditis_elegans.WBcel235.113.gtf
PCGTF=$HERE/data/raw/protein_coding.gtf
REF=$HERE/data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
IDX=/root/aligner-data/star
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
RES=$OUT/timings.jsonl
[ -f "$PCGTF" ] || grep -E '^#|gene_biotype "protein_coding"' "$GTF" > "$PCGTF"

count() {   # count ARMDIR  (counting the arm's aligned.sam)
    local D=$1
    local SAM=$D/aligned.sam
    local COMMON="-p --countReadPairs -s 0 -T 24 -t exon -g gene_id"
    $TI "$NAME.$(basename $D).fc_all"  "$RES" "$D/fc_all.log"  -- $FC $COMMON -a $GTF -o $D/counts_all.txt $SAM || true
    $TI "$NAME.$(basename $D).fc_frac" "$RES" "$D/fc_frac.log" -- $FC $COMMON -O -M --fraction -a $GTF -o $D/counts_frac.txt $SAM || true
    $TI "$NAME.$(basename $D).fc_pc"   "$RES" "$D/fc_pc.log"   -- $FC $COMMON -a $PCGTF -o $D/counts_pc.txt $SAM || true
    $TI "$NAME.$(basename $D).fc_core" "$RES" "$D/fc_core.log" -- $FC $COMMON -R CORE --Rpath $D -a $GTF -o $D/counts_core.txt $SAM || true
}

for ARM in $ARMS; do
    D=$OUT/$ARM; mkdir -p "$D"
    case $ARM in
    truth)
        ln -sf "$TRUTH_SAM" $D/aligned.sam ;;
    gpu)
        $TI "$NAME.gpu.align" "$RES" "$D/align.log" -- $HERE/build_main $REF $D/aligned.sam $R1 $R2 ;;
    star_annot)
        $TI "$NAME.star_annot.align" "$RES" "$D/align.log" -- $STAR --runThreadN 24 --genomeDir $IDX/idx_annot \
            --readFilesIn $R1 $R2 --outFileNamePrefix $D/star_ --outSAMtype SAM --outSAMattributes NH HI AS nM
        mv $D/star_Aligned.out.sam $D/aligned.sam ;;
    star_annot40k)
        $TI "$NAME.star_annot40k.align" "$RES" "$D/align.log" -- $STAR --runThreadN 24 --genomeDir $IDX/idx_annot \
            --readFilesIn $R1 $R2 --outFileNamePrefix $D/star_ --outSAMtype SAM --outSAMattributes NH HI AS nM \
            --alignIntronMax 40000 --alignMatesGapMax 40000
        mv $D/star_Aligned.out.sam $D/aligned.sam ;;
    star_denovo2p)
        $TI "$NAME.star_denovo2p.align" "$RES" "$D/align.log" -- $STAR --runThreadN 24 --genomeDir $IDX/idx_denovo \
            --readFilesIn $R1 $R2 --outFileNamePrefix $D/star_ --outSAMtype SAM --outSAMattributes NH HI AS nM \
            --alignIntronMax 40000 --alignMatesGapMax 40000 --twopassMode Basic
        mv $D/star_Aligned.out.sam $D/aligned.sam ;;
    esac
    [ -n "$NOCOUNT" ] || count "$D"
    [ -z "$CLEAN" ] || rm -rf "$D"/aligned.sam "$D"/star__STARgenome "$D"/star__STARpass1
done
