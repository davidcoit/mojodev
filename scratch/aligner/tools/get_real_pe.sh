#!/bin/bash
# Fetch a real C. elegans paired-end RNA-seq run and convert it to FASTQ.
#   SRR10065383: wild-type N2 control ("N2C1"), Illumina HiSeq 4000, 13.2 M read pairs x 101 bp,
#   from the UV-irradiation RNA-seq study PRJNA563830.
# Usage: tools/get_real_pe.sh [DATA_ROOT]     (run from scratch/aligner)
#
# Notes from doing this once:
#  * ENA's FTP/HTTPS mirror served ~0.5 MB/s here; NCBI's S3 mirror of the .sra gave ~11 MB/s.
#  * The 4.7 GB x 2 FASTQ output needs a big disk. Do not put it on a nearly full mount:
#    fasterq-dump refuses with "disk-limit exceeded" when free space is too small.
set -e
ACC=${ACC:-SRR10065383}
ROOT=${1:-/root/aligner-data}
SRA_TOOLS=data/tools/sratoolkit/sratoolkit.3.4.1-ubuntu64/bin/fasterq-dump
mkdir -p "$ROOT/$ACC/sra" "$ROOT/$ACC/fastq" "$ROOT/$ACC/tmp" data/tools/sratoolkit

if [ ! -x "$SRA_TOOLS" ]; then
    curl -sSfL -o data/tools/sratoolkit/sratoolkit.tar.gz \
        https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz
    tar -xzf data/tools/sratoolkit/sratoolkit.tar.gz -C data/tools/sratoolkit
fi
if [ ! -f "$ROOT/$ACC/sra/$ACC.sra" ]; then
    curl -sSfL -o "$ROOT/$ACC/sra/$ACC.sra" "https://sra-pub-run-odp.s3.amazonaws.com/sra/$ACC/$ACC"
fi
"$SRA_TOOLS" --split-files --threads 16 --temp "$ROOT/$ACC/tmp" -O "$ROOT/$ACC/fastq" "$ROOT/$ACC/sra/$ACC.sra"
ls -la "$ROOT/$ACC/fastq"
