import os
import pickle
import argparse
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from cnn import Net
from dataset import Dataset

# parse arguments
parser = argparse.ArgumentParser(description='Tests CNN')
parser.add_argument('reads_dir', type=str, help='Input dir for long reads')
parser.add_argument('cnn_model_file', type=str, help='Name of model to test')

def main():
	# variables to change
	args = parser.parse_args()
	reads_dir = args.reads_dir
	cnn_model_file = args.cnn_model_file

	cnn_dir = '../data/cnn'
	kmer_length = 50
	max_num_samples_per_species = 1000
	batch_size = 4

	# file locations
	cnn_name = os.path.basename(reads_dir)
	cnn_dir = os.path.join(cnn_dir,cnn_name)
	cnn_model_file = os.path.join(cnn_dir,cnn_model_file)

	# read parameters file to get information of dataset
	param_file = os.path.join(reads_dir,'parameters.txt')
	with open(param_file, 'r') as f:
		read_length = int(f.readline().rstrip().split(': ')[1])

	reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]
	species_list = [file.split('.')[0] for file in reads_files]
	num_species = len(reads_files)
	image_size = read_length - kmer_length + 1

	# retrieve testing parition and labels
	with open(os.path.join(cnn_dir,'test_list.pickle'), 'rb') as f:
		test_list_raw = pickle.load(f)

	with open(os.path.join(cnn_dir,'labels.pickle'), 'rb') as f:
		labels_dict = pickle.load(f)

	test_dict = dict()
	test = []
	cur_label = 0
	for ID in test_list_raw:
		label = labels_dict[ID]
		if label == cur_label:
			test.append(ID)
		if len(test) == max_num_samples_per_species:
			test_dict[cur_label] = test
			test = []
			cur_label += 1

	test_list = sum(test_dict.values(), [])

	print("Testing on " + str(max_num_samples_per_species) + " reads per species")
	print("Testing on " + str(len(test_list)) + " reads total" )

	# generators
	transform = transforms.Compose([transforms.ToTensor()])
	testset = Dataset(reads_dir, reads_files, test_list, labels_dict, kmer_length, transform=transform)
	testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	if use_cuda: torch.backends.cudnn.benchmark = True

	# initialize CNN
	net = Net(image_size, num_species)
	net.load_state_dict(torch.load(cnn_model_file, map_location=device))
	net.to(device)

	# test on testing data
	print("Compute outputs")
	print("---------------")

	class_predict = []
	class_true = []
	class_correct = list(0. for i in range(num_species))
	class_total = list(0. for i in range(num_species))
	with torch.no_grad():
		for local_data in testloader:
			# get samples and labels
			local_batch, local_labels = local_data
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			outputs = net(local_batch.float())
			_, predicted = torch.max(outputs, 1)
			c = (predicted == local_labels).squeeze()
			for i in range(batch_size):
				local_label = local_labels[i]
				class_predict.append(predicted[i].item())
				class_true.append(local_labels[i].item())
				class_correct[local_label] += c[i].item()
				class_total[local_label] += 1


	# sort species by alphabetical order
	sort_index = np.argsort(species_list)
	species_list = [species_list[x] for x in sort_index]
	class_correct = [class_correct[x] for x in sort_index]
	class_total = [class_total[x] for x in sort_index]
	class_predict = [sort_index[y] for y in class_predict]
	class_true = [sort_index[y] for y in class_true]


	print('Accuracy by class')
	for i in range(num_species):
	    print('%5s : %2d %%' % (
	        species_list[i], 100 * class_correct[i] / class_total[i]))

	print('Overall accuracy on test set')
	print('%d %%' % (100 * sum(class_correct) / sum(class_total)))

	print('Confusion Matrix')
	cf_m = confusion_matrix(class_predict, class_true)
	cf_m_file = os.path.join(cnn_dir,'confusion_matrix.' + os.path.basename(cnn_model_file) + '.npy')
	np.save(cf_m_file, cf_m)
	print('saved to', cf_m_file)
	print(cf_m)

if __name__ == "__main__":
	start = datetime.now()
	main()
	end = datetime.now()

	dt_string_start = start.strftime("%d/%m/%Y %H:%M:%S")
	dt_string_end = end.strftime("%d/%m/%Y %H:%M:%S")
	print(dt_string_start)
	print(dt_string_end)
