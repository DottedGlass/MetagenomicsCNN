import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
from Bio import SeqIO
from pyts.image import GramianAngularField

class Dataset(data.Dataset):
	def __init__(self, reads_dir, reads_files, list_IDs, labels, kmer_length, transform=None):
		"""Initialization
		Parameters
	    ----------
	    list_IDs : list
	        list of file names and sample number specific to that file.
			Each element in the list is a string of the form
			reads_file:sample_num
			e.g.
			NC_004113.reads.1000bp.fa:0
		kmer_length : int
			length of kmers used for encoding the reads as time series
		"""

		self.list_IDs = list_IDs
		self.labels = labels
		self.kmer_length = kmer_length
		self.transform = transform

		# open long reads
		long_reads = dict()
		for file in reads_files:
			print("Opening long reads file:",file)
			bioseq_list = list(SeqIO.parse(os.path.join(reads_dir,file),"fasta"))
			reads = np.array([str(bioseq_list[i].seq) for i in range(len(bioseq_list))])
			long_reads[file] = reads
		self.long_reads = long_reads

	def __len__(self):
		"""Denotes total number of samples
		"""
		return len(self.list_IDs)

	def __getitem__(self, idx):
		"""Generates one sample of data
		"""

		# select sample and load label
		ID = self.list_IDs[idx]

		file, sample_num = ID.split(':')
		sample_num = int(sample_num)
		y = self.labels[ID]

		# load read
		read = self.long_reads[file][sample_num]

		# convert read to time series
		ts = self.read2ts(read,self.kmer_length)

		# convert time series to GAF
		gasf = GramianAngularField(method='summation')
		X = gasf.fit_transform(ts.reshape(1,-1)).squeeze()

		# apply transform
		if self.transform:
			X = self.transform(X)

		return X, y

	def read2ts(self, read, kmer_length):
		"""Converts the read into a 1D array of numbers by encoding kmer as numbers
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
		time_series = []
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
