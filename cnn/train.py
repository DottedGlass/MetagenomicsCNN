import os
import pickle
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from pyts.image import GramianAngularField
from cnn import Net
from dataset import Dataset

# parse arguments
parser = argparse.ArgumentParser(description='Trains CNN')
parser.add_argument('reads_dir', type=str, help='Input dir for long reads')

def main():
	# variables to change
	args = parser.parse_args()
	reads_dir = args.reads_dir

	cnn_dir = '../data/cnn'
	kmer_length = 50
	test_percent = 0.2
	max_epochs = 100

	# prepare output folder that will contain the model and train/test split
	cnn_name = os.path.basename(reads_dir)
	cnn_dir = os.path.join(cnn_dir,cnn_name)
	if not os.path.exists(cnn_dir):
		os.makedirs(cnn_dir)

	print("--Model output will be saved in--")
	print(os.path.abspath(cnn_dir))

	# read parameters file to get information of dataset
	param_file = os.path.join(reads_dir,'parameters.txt')
	num_samples = dict()
	with open(param_file, 'r') as f:
		read_length = int(f.readline().rstrip().split(': ')[1])
		error_rate = int(f.readline().rstrip().split(': ')[1])
		coverage = int(f.readline().rstrip().split(': ')[1])
		_ = f.readline()  	#skip this line
		_ = f.readline()	#skip this line
		for line in f:
			species, count = line.split(': ')
			num_samples[species] = int(count)

	reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]
	reads_files.sort() # sort species alphabetically
	num_species = len(reads_files)
	image_size = read_length - kmer_length + 1

	# make training and testing partition
	test_list = []
	train_list = []
	labels_dict = dict()
	for i,file in enumerate(reads_files):
		species = file.split('.')[0]
		num = num_samples[species]
		test_thres = round(num*test_percent)

		# add to test_list
		for k in range(0,test_thres):
			ID = file + ':' + str(k)
			test_list.append(ID)
			labels_dict[ID] = i

		# add to train_list
		for k in range(test_thres,num):
			ID = file + ':' + str(k)
			train_list.append(ID)
			labels_dict[ID] = i

	# save training and testing partition, along with labels
	with open(os.path.join(cnn_dir,'train_list.pickle'), 'wb') as f:
		pickle.dump(train_list, f)

	with open(os.path.join(cnn_dir,'test_list.pickle'), 'wb') as f:
		pickle.dump(test_list, f)

	with open(os.path.join(cnn_dir,'labels.pickle'), 'wb') as f:
		pickle.dump(labels_dict, f)

	# generators
	transform = transforms.Compose([transforms.ToTensor()])
	trainset = Dataset(reads_dir, reads_files, train_list, labels_dict, kmer_length, transform=transform)
	trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	if use_cuda: torch.backends.cudnn.benchmark = True

	# initialize CNN
	net = Net(image_size, num_species)
	net.to(device)

	# define loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# loop over epochs
	for epoch in range(max_epochs):
		running_loss = 0.0
		# training
		for i, local_data in enumerate(trainloader):
			# get samples and labels
			local_batch, local_labels = local_data
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			optimizer.zero_grad()
			outputs = net(local_batch.float())
			loss = criterion(outputs, local_labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 2000 == 1999:
				now = datetime.now()
				dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
				print('%s [%d, %5d] loss: %.3f' % (dt_string, epoch+1, i+1, running_loss / 2000))
				running_loss = 0.0

			# save progress after every 10000 batches
			if i % 10000 == 9999:
				cnn_save_name = os.path.join(cnn_dir, "cnn_epoch_" + str(epoch) + ".i_" + str(i) + ".pth")
				torch.save(net.state_dict(), cnn_save_name)
				print('Saved:',cnn_save_name)


	cnn_save_name_final = os.path.join(cnn_dir, "cnn_final.pth")
	torch.save(net.state_dict(), cnn_save_name_final)
	print('Finished training. Model path is')
	print(os.path.abspath(cnn_save_name_final))

if __name__ == "__main__":
	main()
