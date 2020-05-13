import os
import matplotlib.pyplot as plt

dump_list = os.listdir('./dump')

kmer_dict = dict()
for dump in dump_list:
	path = './dump/' + dump
	kmer_set = set()
	f = open(path, 'r')
	for line in f:
		if line[0] == '>':
			continue
		kmer_set.add(line.strip())
	kmer_dict[dump] = kmer_set
	f.close()

# find unique kmers for each genome
genome_label = []
unique_count = []
for dump in dump_list:
	temp = dump_list.copy()
	temp.remove(dump)
	temp_set = kmer_dict[dump].copy()
	for t in temp:
		temp_set = temp_set - kmer_dict[t]
	dump_name = dump.split('_')
	genome_label.append('_'.join(dump_name[:2]))
	unique_count.append(len(temp_set))

fig, axes = plt.subplots(tight_layout=True)
bar = axes.bar(genome_label, unique_count)
axes.set_xlabel('Genome Name')
axes.set_ylabel('Unique 50-mer count')
plt.xticks(rotation=45)
plt.savefig('figure.png')
