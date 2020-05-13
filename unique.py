import os
import matplotlib.pyplot as plt

dump_list = os.listdir('./dump')

dict_list = []
for dump in dump_list:
	path = './dump/' + dump
	f = open(path, 'r')
	dump_dict = dict()
	is_kmer = False
	count = 0
	for line in f:
		if not is_kmer:
			count = int(line.strip()[1:])
		else:
			dump_dict[line.strip()] = count
		is_kmer = is_kmer == False
	f.close()
	dict_list.append(dump_dict)
print(dict_list)
