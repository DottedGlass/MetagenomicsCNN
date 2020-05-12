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

    num_kmers = len(read) - kmer_length + 1
    nt2int = {'A':'0', 'C':'1', 'G':'2', 'T':'3'}
    time_series = np.zeros(len(num_kmers),dtype=int)
    for i in range(num_kmers):

        # split into kmers
        kmer = read[i:i+kmer_length]

        # convert to base 4 num
        int_mer  = []
        for b in kmer:
            if b not in nt2int:
                b = 'A'         # replace unknown bases with A
            int_mer.append(nt2int[b])

        kmer_base4 = "".join(int_mer)

        # convert to base 10 number
        kmer_num = int(kmer_base4,base=4)

        time_series.append(kmer_num)

    return np.array(time_series)

def onehot_encode(read):
    """Converts the read into 1 hot encoding. Returns a (len(read),4) array.
    Col 1 corresponds to A
    Col 2 corresponds to C
    Col 3 corresponds to G
    Col 4 corresponds to T

    Parameters
    ----------
    read : str
        A read from shotgun sequencing. Bases must be A, C, G, or T.

    Returns
    -------
    numpy.array
    """

    onehot = np.zeros((len(read),4))

    for i, bp in enumerate(read):
        if bp == 'A': onehot[i,0] = 1
        if bp == 'C': onehot[i,1] = 1
        if bp == 'G': onehot[i,2] = 1
        if bp == 'T': onehot[i,3] = 1

    return onehot

## TODO: implement piecewise aggregation approximation to smooth the time
# series, reducing size of GAF output matrix

# TODO: implement MTF

if __name__ == "__main__":
    read = sys.argv[1]
    print(read2array(read,kmer_length=3))
