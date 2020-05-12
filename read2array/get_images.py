"""Driver script to convert all reads in reads directory into images

Usage: python get_images.py ../data/long_reads/ ../data/images/

Reads belonging to a species are saved as 3D array.
(read#, read_length, read_length)

e.g.
Species A has 300 reads of length 150. Assuming 10-mers, after running the GAF
transform, each corresponding image is 141x141.
This script will save the 300 images as (141,141) numpy arrays
"""

import os
import argparse
import numpy as np
from Bio import SeqIO
from pyts.image import GramianAngularField
from read2array import read2num


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Convert reads into images')
    parser.add_argument('indir', type=str, help='Input dir for reads')
    parser.add_argument('outdir', type=str, help='Output dir for reads encoded as images')
    parser.add_argument('--kmer_length', default=100, help='Length of kmers used for encoding read into numbers')
    parser.add_argument('--array_type', default='GAF', help="Specifies type of array the read will be encoded in. 'GAF' denotes Gramian Angular Field 'MTF' denotes Markov Transition Fields")
    args = parser.parse_args()

    kmer_length = int(args.kmer_length)
    reads_dir = args.indir
    images_dir = args.outdir + '/' + os.path.basename(reads_dir) + '/'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # get list of fasta files that contain the reads
    reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]

    # TODO: Parallelize this for loop
    print("processing files")
    print("----------------")
    print(reads_files)
    print("----------------")
    for f in reads_files:
        print(f)

        # read in reads as list
        reads = list(SeqIO.parse(os.path.join(reads_dir,f),"fasta"))

        # convert reads into numeric encoding
        print("Encoding reads as time series")
        numeric_reads = []
        for i in range(len(reads)):
            numeric_reads.append(read2num(str(reads[i].seq), kmer_length=kmer_length))

        numeric_reads = np.array(numeric_reads)

        # GAF conversion
        print("Converting time series into image")
        gasf = GramianAngularField(method='summation')
        gaf = gasf.fit_transform(numeric_reads)

        array_file_name = os.path.splitext(f)[0]
        array_path = os.path.join(images_dir, array_file_name + ".npy")
        np.save(array_path, gaf)

    print('Images saved in')
    print(os.path.abspath(images_dir))
