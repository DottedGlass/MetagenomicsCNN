import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from cnn import Net
import os


PATH = '../data/images'
TEST_SIZE = 10
image_size = 141

transform = transforms.Compose([transforms.ToTensor()])

files = os.listdir(PATH)
classes = [f.split('.'[0]) for f in files]

# retrieve the data from each file
# except the first 10, load the rest
class TrainSet(torch.utils.data.Dataset):
	def __init__(self, filepath, test_size, transform=None):
		self.data = []
		self.transform = transform
		files = os.listdir(filepath)
		for i, f in enumerate(files):
			dat = np.load(filepath + '/' + f)[test_size:]
			for d in dat:
				t = d.reshape(image_size, -1)
				self.data.append( (t, i) )

	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample, label = self.data[idx]
		if self.transform:
			sample = self.transform(sample)
		return sample, label

trainset = TrainSet(PATH, TEST_SIZE, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

net = Net()

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
			running_loss = 0.0
print('Finished training')

torch.save(net.state_dict(), PATH)
