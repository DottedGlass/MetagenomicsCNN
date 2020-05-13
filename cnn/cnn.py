import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

class Net(nn.Module):
	def __init__(self, img_size, output_size):
		super(Net, self).__init__()
		# initialize params
		self.img_size = img_size
		conv1_out, conv1_ksize = (6, 5)
		pool_size, pool_stride = (2, 2)
		conv2_out, conv2_ksize = (16, 5)
		fc1_out = 120
		fc2_out = 84

		# double conv2d + maxpool
		self.conv1 = nn.Conv2d(1, conv1_out, conv1_ksize) # 1 in, 6 out, 5 kernel size
		self.pool = nn.MaxPool2d(pool_size, pool_stride)
		self.conv2 = nn.Conv2d(conv1_out, conv2_out, conv2_ksize)

		# calculate input size for fc1
		self.fc1_in = conv2d_pool_size(self.img_size, conv1_ksize, pool_size, pool_stride)
		self.fc1_in = conv2d_pool_size(self.fc1_in, conv2_ksize, pool_size, pool_stride)
		self.fc1_in = self.fc1_in * self.fc1_in * conv2_out

		# 3 fully connected layers
		self.fc1 = nn.Linear(self.fc1_in, fc1_out)
		self.fc2 = nn.Linear(fc1_out, fc2_out)
		self.fc3 = nn.Linear(fc2_out, output_size)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		# print(x.shape)
		x = self.pool(F.relu(self.conv2(x)))
		# print(x.shape)
		x = x.view(-1, self.fc1_in)
		# print(x.shape)
		x = F.relu(self.fc1(x))
		# print(x.shape)
		x = F.relu(self.fc2(x))
		# print(x.shape)
		x = self.fc3(x)
		# print(x.shape)
		# print("Done")
		return x

def conv2d_pool_size(img_size, conv_ksize, pool_size, pool_stride):
	return math.floor((img_size-conv_ksize+1-pool_size)/pool_stride + 1)
