import torch as t
import os
from PIL import Image
from torch.autograd import Variable as V
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,models
from torch.utils import data
from torchvision import transforms as T
from torch.nn import functional as F
import pickle as pk
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch import nn
###############Data######################################
f=open('RML2016.10a_dict.pkl','rb')
data=pk.load(f,encoding='latin1')
#print(data)
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(data[(mod,snr)])
        for i in range(data[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,int(n_examples)), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])  #(2,128)
#print (X_train.shape, in_shp)   #(11000,2,128)   (2,128)
classes = mods
X_train=t.from_numpy(X_train)
X_train=X_train.unsqueeze(1)
Y_train=t.from_numpy(Y_train)
X_test=t.from_numpy(X_test)
X_test=X_test.unsqueeze(1)
Y_test=t.from_numpy(Y_test)
###############Net####################################
model=nn.Sequential(
		nn.Conv2d(1,256,kernel_size=(1,3),padding=(0,2)),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Conv2d(256,80,kernel_size=(2,3),padding=(0,2)),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Flatten(),
		
		nn.Linear(10560,256),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Linear(256,11),
		nn.Softmax(0))
model=model.cuda()
# model=models.vgg19(pretrained=True)
# for param in model.parameters():
    # param.requires_grad = False
# model = model.cuda()
# model.classifier[6] = Sequential(Linear(4096, 11))
# for param in model.classifier[6].parameters():
    # param.requires_grad = True
    
################Train############################
nb_epoch = 15     # number of epochs to train on
batch_size = 128  # training batch size

criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

t.manual_seed(1)
label = t.argmax(Y_train, -1)  #One-hot to label
dataset = Data.TensorDataset(X_train,label) #TensorData Type
loader = Data.DataLoader(dataset = dataset, batch_size =128, shuffle = False)
print('Begin!!!')
try:
	for epoch in range(0,nb_epoch):
		running_loss=0.0
		acc_num=0
		for i,data in enumerate(loader): #枚举
			inputs,labels=data
			inputs,labels=V(inputs),V(labels)
			inputs,labels=inputs.cuda(),labels.cuda()
			optimizer.zero_grad()
			#forward+backward
			outputs=model(inputs)
			labels=labels.squeeze(-1); #[128,1]转为[128]一维
			loss=criterion(outputs,labels)
			loss.backward()
			optimizer.step()
	
			running_loss+=loss
			if i%43==0:
				print(str([epoch+1,i+1])+' loss:'+str(running_loss/43))#平均损失函数值
				_,pred=t.max(outputs,1)
				num_correct=(pred==labels).sum()
				acc=int(num_correct)/128
				print('Acc:'+str(acc))
				acc=0.0
				running_loss=0.0
except KeyboardInterrupt:
	PATH = './Radio_VGG19.pth'
	t.save(model.state_dict(), PATH)
	f=open('acc.txt','w')
	for i in acc_plot:
		f.write(str(i)+'\n')
	f.close()
	f=open('loss.txt','w')
	for i in loss_plot:
		f.write(str(i)+'\n')
	f.close()
print('Done!')


PATH = './Radio_VGG19.pth'
t.save(model.state_dict(), PATH)
f=open('acc.txt','w')
for i in acc_plot:
	f.write(str(i)+'\n')
	f.close()
f=open('loss.txt','w')
for i in loss_plot:
	f.write(str(i)+'\n')
	f.close()
