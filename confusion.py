import matplotlib.pyplot as plt
import numpy as np

cm_dir = './cnn/confusion_matrices/'
cm_list = ['read_500_error_1_confusion_matrix.cnn_epoch_3.i_289999.pth.npy',
					 'read_500_error_10_confusion_matrix.cnn_epoch_3.i_219999.pth.npy',
					 'read_1000_error_1_confusion_matrix.cnn_epoch_1.i_219999.pth.npy',
					 'read_1000_error_10_confusion_matrix.cnn_epoch_1.i_49999.pth.npy']

label1 = ['NC_004113','NC_020246','NC_023013','NC_014374','NC_014494','NC_008698','NC_009515','NC_018621','NZ_LN832404','NC_010117']
label2 = ['NC_004113','NC_020246','NC_014374','NC_023013','NC_014494','NC_009515','NC_008698','NZ_LN832404','NC_018621','NC_010117']
label3 = ['NC_010117','NC_023013','NC_014374','NC_009515','NC_018621','NC_008698','NC_020246','NZ_LN832404','NC_004113','NC_014494']
label4 = ['NC_020246','NC_008698','NC_014494','NC_004113','NZ_LN832404','NC_009515','NC_018621','NC_010117','NC_023013','NC_014374']
labels = [label1, label2, label3, label4]
for e, cm in enumerate(cm_list):
	x = np.load(cm_dir + cm)
	indexes = [label1.index(k) for k in labels[e]]
	print('before')
	print(x)
	x = x[indexes]
	for k, y in enumerate(x):
		x[k] = y[indexes]
	print('after')
	print(x)
	fig, ax = plt.subplots()
	im = ax.imshow(x)
	ax.set_xticks(np.arange(len(label1)))
	ax.set_yticks(np.arange(len(label1)))
	ax.set_xticklabels(label1)
	ax.set_yticklabels(label1)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
	for i in range(len(label1)):
		for j in range(len(label1)):
			text = ax.text(j, i, x[i, j], ha="center", va="center", color="w")
	ax.set_title("Confusion Matrix")
	fig.tight_layout()
	plt.savefig(cm + '.png')