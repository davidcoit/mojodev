#!/bin/bash
# Fetch and build everything the STAR / featureCounts comparison needs, in isolated directories
# under data/tools/ (nothing is installed system-wide).   Run from scratch/aligner.
#   STAR 2.7.11b       prebuilt static binary
#   zlib 1.3.1         built into data/tools/zlib/install (the system has no zlib headers)
#   Subread 2.0.6      featureCounts BUILT FROM SOURCE: the prebuilt static binary segfaults at
#                      startup in this container
#   STAR indexes       annotated and de novo, in /root/aligner-data/star (outside the repo)
set -e
T=data/tools
mkdir -p $T/star $T/zlib $T/subread-src
[ -f $T/star/STAR.zip ] || curl -sSfL -o $T/star/STAR.zip \
    https://github.com/alexdobin/STAR/releases/download/2.7.11b/STAR_2.7.11b.zip
[ -d $T/star/STAR_2.7.11b ] || python3 -I -c "import zipfile; zipfile.ZipFile('$T/star/STAR.zip').extractall('$T/star')"
chmod +x $T/star/STAR_2.7.11b/Linux_x86_64_static/STAR
[ -f $T/zlib/zlib.tar.gz ] || curl -sSfL -o $T/zlib/zlib.tar.gz \
    https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
[ -f $T/subread-src/subread-src.tar.gz ] || curl -sSfL -o $T/subread-src/subread-src.tar.gz \
    "https://sourceforge.net/projects/subread/files/subread-2.0.6/subread-2.0.6-source.tar.gz/download"
[ -x $T/featurecounts-bin/featureCounts ] || bash tools/build_subread.sh
[ -d /root/aligner-data/star/idx_annot ] || bash tools/star_index.sh
$T/star/STAR_2.7.11b/Linux_x86_64_static/STAR --version
$T/featurecounts-bin/featureCounts -v 2>&1 | grep featureCounts
