"""Driver script to convert all reads in reads directory into time series

Example of usage:
python get_time_series.py ../data/long_reads/read_1000_error_1 ../data/time_series

Reads belonging to a species are saved as 3D array.
(read#, read_length, read_length)

e.g.
Species A has 300 reads of length 150. Assuming 10-mers, the corresponding
time series has length 141.
This script will save the 300 time series numpy arrays
"""

import os
import argparse
from joblib import Parallel, delayed
import numpy as np
from Bio import SeqIO
# from pyts.image import GramianAngularField
from read2array import read2num

def save_ts(read_file,reads_dir,ts_dir,kmer_length):
    """ Converts reads in read_file to time series
    """
    species = read_file.split('.')[0]
    # read in reads as list
    bioseq_list = list(SeqIO.parse(os.path.join(reads_dir,read_file),"fasta"))
    reads = np.array([str(bioseq_list[i].seq) for i in range(len(bioseq_list))])

    # loop over all reads
    time_series_list = []
    for i in range(len(reads)):
        ts = read2num(reads[i], kmer_length=kmer_length)
        time_series_list.append(ts)

        if i % 5000 == 4999:
            print("processed read", str(i+1), "in", species)

    time_series = np.array(time_series_list)

    # save file
    ts_path = os.path.join(ts_dir, species + '.npy')
    np.save(ts_path, time_series)



if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Convert reads into time series')
    parser.add_argument('indir', type=str, help='Input dir for reads')
    parser.add_argument('outdir', type=str, help='Output dir for reads encoded as time series')
    parser.add_argument('--kmer_length', default=50, help='Length of kmers used for encoding read into numbers')
    parser.add_argument('--cpu', default=1, help='Number of cpu cores used for parallel processing')
    args = parser.parse_args()

    kmer_length = int(args.kmer_length)
    reads_dir = args.indir
    ts_dir = args.outdir + '/' + os.path.basename(reads_dir) + '/'
    if not os.path.exists(ts_dir):
        os.makedirs(ts_dir)

    # get number of cores for parallel processing
    num_cores = int(args.cpu)

    # get list of fasta files that contain the reads
    reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]

    # process files
    print("processing files")
    print("----------------")
    print(reads_files)
    print("----------------")

    Parallel(n_jobs=num_cores)(delayed(save_ts)(f,reads_dir,ts_dir,kmer_length) for f in reads_files)

    print('Time series saved in')
    print(os.path.abspath(ts_dir))
