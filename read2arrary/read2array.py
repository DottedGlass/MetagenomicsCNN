"""Converts 1D reads from shotgun sequencing into 2D arrays.

Implements encodings described in:

Wang, Zhiguang, and Tim Oates. "Encoding time series as images for visual
inspection and classification using tiled convolutional neural networks." In
Workshops at the Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
"""

import sys
import numpy as np

def read2array(read, array_type='GAF'):
    """Converts a string into an array.

    Parameters
    ----------
    read : str
        A read from shotgun sequencing. Bases must be A, C, G, or T.
    array_type : 'GAF' or 'MTF'
        Specifies type of array the read will be encoded in.
        'GAF' denotes Gramian Angular Field
        'MTF' denotes Markov Transition Fields

    Returns
    -------
    numpy.array
    """

    # encode read as 1D array of integers (time series)
    nt2int = {'A':0, 'C':1, 'G':2, 'T':3}
    time_series = np.array([nt2int[b] for b in read])

    if array_type == 'GAF':
        array = getGAF(time_series)

    return array


def getGAF(x):
    """Converts time series into GAF

    Parameters
    ----------
    X : numpy.array
        1D numpy array with integers encoding the nucleotide bases

    Returns
    -------
    numpy.array
        GAF matrix
    """

    # rescale X to be [-1,1]
    x_max = np.max(x)
    x_min = np.min(x)

    x_scal = (2*x - x_max - x_min) / (x_max - x_min)

    # polar encoding
    phi = np.arccos(x_scal)

    # compute GAF with phi
    N = len(phi)
    GAF = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            GAF[i,j] = np.cos(phi[i]+phi[j])

    return GAF

# TODO: encode k-mers into integers intead of 1-mers
# e.g. 2-mers can be encoded into 16 integers

## TODO: implement piecewise aggregation approximation to smooth the time
# series, reducing size of GAF output matrix

# TODO: implement MTF

if __name__ == "__main__":
    read = sys.argv[1]
    print(read2array(read))
