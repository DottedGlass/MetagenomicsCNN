import os
import argparse
import numpy as np
from Bio import SeqIO
from pyts.image import GramianAngularField
from read2array import read2num


"""Driver script to convert all reads in reads directory into images

Usage: python get_images.py ../data/mason_reads ../data/images

Reads belonging to a species are saved as 3D array.
(read#, read_length, read_length)

e.g.
Species A has 300 reads of length 150. Assuming 10-mers, after running the GAF
transform, each corresponding image is 141x141.
This script will save the 300 images as a (300,141,141) numpy array
"""

# parse arguments
parser = argparse.ArgumentParser(description='Convert reads into images')
parser.add_argument('indir', type=str, help='Input dir for reads')
parser.add_argument('outdir', type=str, help='Output dir for reads encoded as images')
parser.add_argument('--kmer_length', default=10, help='Length of kmers used for encoding read into numbers')
parser.add_argument('--array_type', default='GAF', help="Specifies type of array the read will be encoded in. 'GAF' denotes Gramian Angular Field 'MTF' denotes Markov Transition Fields")
args = parser.parse_args()

reads_dir = args.indir
images_dir = args.outdir + '/' + os.path.basename(reads_dir) + '/'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# get list of fasta files that contain the reads
reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]

# TODO: Parallelize this for loop
for f in reads_files:
    print(f)

    reads = list(SeqIO.parse(os.path.join(reads_dir,f),"fasta"))

    nrow = len(reads)
    num_kmers = len(reads[0].seq) - args.kmer_length + 1

    # convert reads into numeric encoding
    numeric_reads = []
    for i in range(len(reads)):
        numeric_reads.append(read2num(str(reads[i].seq), kmer_length=args.kmer_length))

    numeric_reads = np.array(numeric_reads)

    # GAF conversion
    gasf = GramianAngularField(method='summation')
    gaf = gasf.fit_transform(numeric_reads)

    array_file_name = os.path.splitext(f)[0]
    array_path = os.path.join(images_dir, array_file_name + ".npy")
    np.save(array_path, gaf)

print('Images saved in')
print(os.path.abspath(images_dir))
