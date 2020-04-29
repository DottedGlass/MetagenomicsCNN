import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint
from Bio import SeqIO
from read2array import read2num


# read in first two reads from NC_008245
reads_file = '../data/mason_reads/NC_008245.reads.150.fa'
kmer_length = 10

reads = list(SeqIO.parse(reads_file,"fasta"))
reads = reads[0:2]
X = []

for i in range(len(reads)):

    time_series = read2num(str(reads[i].seq),kmer_length=kmer_length)
    X.append(time_series)

X = np.array(X)

# Transform the time series into Gramian Angular Fields
gasf = GramianAngularField(method='summation')
X_gasf = gasf.fit_transform(X)

# Show the images for the first time series
fig = plt.figure()
plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
plt.title('Gramian Angular Field')
plt.colorbar()
plt.show()
