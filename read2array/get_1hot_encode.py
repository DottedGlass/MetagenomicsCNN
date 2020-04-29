import os
import argparse
import numpy as np
from Bio import SeqIO
from read2array import onehot_encode


"""Driver script to convert all reads in reads directory into one-hot encoded format

Usage: python get_1hot_encode.py ../data/long_reads/ ../data/onehot
"""

# parse arguments
parser = argparse.ArgumentParser(description='One-hot encoding of reads')
parser.add_argument('indir', type=str, help='Input dir for reads')
parser.add_argument('outdir', type=str, help='Output dir for reads to be one-hot encoded')
args = parser.parse_args()

reads_dir = args.indir
out_dir = args.outdir + '/' + os.path.basename(reads_dir) + '/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# get list of fasta files that contain the reads
reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]

# TODO: Parallelize this for loop
print("processing files")
print("----------------")
print(reads_files)
print("----------------")
for f in reads_files:
    print(f)

    reads = list(SeqIO.parse(os.path.join(reads_dir,f),"fasta"))

    # 1 hot encoding
    onehot = []
    for i in range(len(reads)):
        onehot.append(onehot_encode(str(reads[i].seq)))

    onehot = np.array(onehot)

    array_file_name = os.path.splitext(f)[0]
    array_path = os.path.join(out_dir, array_file_name + ".npy")
    np.save(array_path, onehot)

print('One-hot encoding saved in')
print(os.path.abspath(out_dir))
