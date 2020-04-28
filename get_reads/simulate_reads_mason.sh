#!/bin/bash

# list of references to simulate reads from
declare -a StringArray=("NC_010117" "NZ_LN832404" "NC_018621" "NC_014494" "NC_004113" "NC_009515" "NC_023013" "NC_008698" "NC_020246" "NC_014374")

# length of reads
LENGTH=150

# output folder
OUTDIR=mason_reads

# simulate reads using mason
mkdir -p ../data/${OUTDIR}
for val in ${StringArray[@]}; do
   mason_simulator -ir ../data/RefSeq/${val}.fasta -n 10000 --illumina-read-length $LENGTH -o ../data/${OUTDIR}/${val}.reads.${LENGTH}.fa -oa ../data/${OUTDIR}/${val}.alignments.sam
done
