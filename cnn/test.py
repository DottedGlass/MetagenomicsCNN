import os
import pickle
import argparse
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

# variables to change
reads_dir = '/home-4/xwang145@jhu.edu/workzfs-mschatz1/xwang145/data/long_reads/read_1000_error_1'
cnn_dir = '/work-zfs/mschatz1/xwang145/data/cnn'
cnn_model_file = 'cnn_epoch_0.i_109999.pth'
kmer_length = 50

# file locations
cnn_name = os.path.basename(reads_dir)
cnn_dir = os.path.join(cnn_dir,cnn_name)
cnn_model_file = os.path.join(cnn_dir,cnn_model_file)

# read parameters file to get information of dataset
param_file = os.path.join(reads_dir,'parameters.txt')
with open(param_file, 'r') as f:
	read_length = int(f.readline().rstrip().split(': ')[1])

reads_files = [f for f in os.listdir(reads_dir) if f.endswith('.fa')]
num_classes = len(reads_files)
image_size = read_length - kmer_length + 1

# retrieve testing parition and labels
with open(os.path.join(cnn_dir,'test_list.pickle'), 'rb') as f:
	test_list = pickle.load(f)
with open(os.path.join(cnn_dir,'labels.pickle'), 'rb') as f:
	labels_dict = pickle.load(f)

# generators
transform = transforms.Compose([transforms.ToTensor()])
testset = Dataset(reads_dir, reads_files, test_list, labels_dict, kmer_length, transform=transform)
testloader = data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# # CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# if use_cuda: torch.backends.cudnn.benchmark = True

# initialize CNN
net = Net(image_size, num_classes)
# net.load_state_dict(torch.load(cnn_model_file, map_location=device))
# net.to(device)
net.load_state_dict(torch.load(cnn_model_file))

correct = 0
total = 0
label_list = []
label_correct = []
predicted_list = []
with torch.no_grad():
	for local_data in testloader:
		# get samples and labels
		local_batch, local_labels = local_data
		# Transfer to GPU
		# local_batch, local_labels = local_batch.to(device), local_labels.to(device)

		outputs = net(local_batch.float())
		_, predicted = torch.max(outputs.data, 1)
		total += local_labels.size(0)
		correct += (predicted == local_labels).sum().item()

		label_list.append(local_labels)
		predicted_list.append(predicted)
		label_correct.append((predicted == local_labels).sum().item())

print('Accuracy of the network on the test reads: %d %%' % (100 * correct / total))

print('Accuracy by label')
for l,c in zip(label_list,label_correct):
	print(l,":",str(c/len(test_list)))

print('Confusion Matrix')
cf_m = confusion_matrix(label_list, predicted_list)
print(cf_m)
