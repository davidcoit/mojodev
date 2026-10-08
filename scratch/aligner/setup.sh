#!/bin/bash
# One-time setup: a venv whose Mojo has the GPU modules (`max.gpu`), the C. elegans
# reference, and a prebuilt minimap2 for the CPU baseline.  Run from scratch/aligner.
set -e
VENV=${VENV:-/root/venvs/aligner}

if [ ! -x "$VENV/bin/mojo" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q max==26.6.0   # pulls mojo 1.1.0 + the max.gpu Mojo package
fi

mkdir -p data/raw data/sim data/tools/minimap2 results
ENS=https://ftp.ensembl.org/pub/release-113
if [ ! -f data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa ]; then
    curl -sSfL -o data/raw/genome.fa.gz "$ENS/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz"
    gunzip -c data/raw/genome.fa.gz > data/raw/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa
fi
if [ ! -f data/raw/Caenorhabditis_elegans.WBcel235.113.gtf ]; then
    curl -sSfL -o data/raw/annotation.gtf.gz "$ENS/gtf/caenorhabditis_elegans/Caenorhabditis_elegans.WBcel235.113.gtf.gz"
    gunzip -c data/raw/annotation.gtf.gz > data/raw/Caenorhabditis_elegans.WBcel235.113.gtf
fi
if [ ! -x data/tools/minimap2/minimap2-2.28_x64-linux/minimap2 ]; then
    curl -sSfL -o data/tools/minimap2/mm2.tar.bz2 https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28_x64-linux.tar.bz2
    tar -xjf data/tools/minimap2/mm2.tar.bz2 -C data/tools/minimap2
fi
echo "setup done"
