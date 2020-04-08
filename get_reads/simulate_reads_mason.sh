#!/bin/bash

# list of references to simulate reads from
declare -a StringArray=("NZ_CP009633" "NC_017449" "NC_017450" "NZ_CP009607" "NZ_CP009353" "NZ_CP010446" "NC_008245" "NC_009257" "NC_017453" "NZ_CP012372")

# simulate reads using mason
for val in ${StringArray[@]}; do
   mason_simulator -ir ../data/RefSeq/${val}.fasta -n 10000 --illumina-read-length 150 -o ../data/mason_reads/${val}.illuminareads.fa -oa ../data/mason_reads/${val}.alignments.sam
done
