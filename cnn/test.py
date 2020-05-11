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
from sklearn.metrics import confusion_matrix

PATH = '../data/images/read_500_error_0.01'
TEST_SIZE = 20
CNN_FILE = './genomics_cnn_final.pth'
transform = transforms.Compose([transforms.ToTensor()])
image_size = 451
output_size = 10

files = os.listdir(PATH)
classes = [f.split('.'[0]) for f in files]

# retrieve the data from each file
# except the first 20, load the rest
class TestSet(torch.utils.data.Dataset):
	def __init__(self, filepath, test_size, transform=None):
		self.data = []
		self.transform = transform
		files = os.listdir(filepath)
		for i, f in enumerate(files):
			dat = np.load(filepath + '/' + f)[:test_size]
			for d in dat:
				self.data.append( (d, i) )

	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample, label = self.data[idx]
		if self.transform:
			sample = self.transform(sample)
		return sample, label

testset = TestSet(PATH, TEST_SIZE, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

net = Net(image_size, output_size)
net.load_state_dict(torch.load(CNN_FILE))

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
label_list = []
label_correct = []
predicted_list = []
with torch.no_grad():
	for data in testloader:
		inputs, labels = data
		outputs = net(inputs.float())
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

		label_list.append(labels)
		predicted_list.append(predicted)
		label_correct.append((predicted == labels).sum().item())

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

print('Accuracy by label')
for l,c in zip(label_list,label_correct):
	print(l,":",str(c/TEST_SIZE))

print('Confusion Matrix')
cf_m = confusion_matrix(label_list, predicted_list)
print(cf_m)
