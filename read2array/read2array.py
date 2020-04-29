"""Converts 1D reads from shotgun sequencing into 2D arrays.

Implements encodings described in:

Wang, Zhiguang, and Tim Oates. "Encoding time series as images for visual
inspection and classification using tiled convolutional neural networks." In
Workshops at the Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
"""

import sys
import numpy as np
from pyts.image import GramianAngularField

def read2array(read, kmer_length=1, array_type='GAF'):
    """Converts a string into an array.

    Parameters
    ----------
    read : str
        A read from shotgun sequencing. Bases must be A, C, G, or T.
    kmer_length: int
        Length of kmers used for encoding read into base
    array_type : 'GAF' or 'MTF'
        Specifies type of array the read will be encoded in.
        'GAF' denotes Gramian Angular Field
        'MTF' denotes Markov Transition Fields

    Returns
    -------
    numpy.array
    """

    # encode read as 1D array of integers (time series)
    time_series = read2num(read,kmer_length)

    if array_type == 'GAF':
        GAF = GramianAngularField(method='summation')
        array = GAF.fit_transform(time_series)

    return array

def read2num(read,kmer_length):
    """Converts the read into a 1D array of numbers
    Parameters
    ----------
    read : str
        A read from shotgun sequencing. Bases must be A, C, G, or T.
    kmer_length: int
        Length of kmers used for encoding read into numbers

    Returns
    -------
    numpy.array
    """

    # remove unknown bases with A
    read = read.replace('N','A')
    read = read.replace('K','A')
    read = read.replace('R','A')
    read = read.replace('Y','A') ## TODO: need to generalize to all unknown bases

    num_kmers = len(read) - kmer_length + 1
    nt2int = {'A':'0', 'C':'1', 'G':'2', 'T':'3'}
    time_series = []
    for i in range(num_kmers):

        # split into kmers
        kmer = read[i:i+kmer_length]

        # convert to base 4 num
        kmer_base4 = "".join([nt2int[b] for b in kmer])

        # convert to base 10 number
        kmer_num = int(kmer_base4,base=4)

        time_series.append(kmer_num)

    return np.array(time_series)

# TODO: encode k-mers into integers intead of 1-mers
# e.g. 2-mers can be encoded into 16 integers

## TODO: implement piecewise aggregation approximation to smooth the time
# series, reducing size of GAF output matrix

# TODO: implement MTF

if __name__ == "__main__":
    read = sys.argv[1]
    print(read2array(read,kmer_length=3))
