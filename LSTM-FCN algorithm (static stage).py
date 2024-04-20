#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset,DataLoader
import numpy as np
from scipy import interpolate
import random
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    #device = torch.device("cpu")
else:
    device = torch.device("cpu")
#single static stage input data 
os.chdir('/kaggle/input/info-combine-0228')
m=900
Nnn=800+800#完整应该是880
heat_up_number=1
Nnn1=800
XX_1= np.array(pd.read_csv('2fracture_T_heatup_multistage_New_type_geo_initial_for_0228.csv'))
y_mm_1=np.array(pd.read_csv('position_heatup_train_2frac_iso_0228.csv'))
real_position_1=np.array(pd.read_csv('2fracture_multistage_realposition_New_type_geo_initial_for_0228.csv'))
XX_1=XX_1.reshape(Nnn1,m+1,heat_up_number)
XX_1=XX_1.transpose(0,2,1)#(number_samples,heatup_stage,m=901)
XX_1=XX_1.reshape(Nnn1*heat_up_number,m+1)
real_position_1=real_position_1.reshape(Nnn1,m+1,heat_up_number)
real_position_1=real_position_1.transpose(0,2,1)#(number_samples,heatup_stage,m=901)
real_position_1=real_position_1.reshape(Nnn1*heat_up_number,m+1)
Nnn2=800
os.chdir("/kaggle/input/iso1219")
XX_2= np.array(pd.read_csv('2fracture_T_heatup_multistage_1217.csv'))
y_mm_2=np.array(pd.read_csv('position_heatup_train_2frac_iso_1217.csv'))
real_position_2=np.array(pd.read_csv('2fracture_multistage_realposition_1217.csv'))
XX_2=XX_2.reshape(Nnn2,m+1,10)[:,:,-2:-1]
XX_2=XX_2.transpose(0,2,1)#(number_samples,heatup_stage,m=901)
XX_2=XX_2.reshape(Nnn2*heat_up_number,m+1)
real_position_2=real_position_2.reshape(Nnn2,m+1,10)[:,:,-2:-1]
real_position_2=real_position_2.transpose(0,2,1)#(number_samples,heatup_stage,m=901)
real_position_2=real_position_2.reshape(Nnn2*heat_up_number,m+1)

XX=np.zeros((Nnn-2,heat_up_number,m+1))
XX=torch.cat((torch.Tensor(XX_1),torch.Tensor(XX_2)),dim=0)
XX=XX.numpy()
XX=XX.reshape((Nnn)*heat_up_number,m+1)
XX_new=np.zeros((Nnn*heat_up_number,m+1))
real_position=np.zeros((Nnn,heat_up_number,m+1))
real_position=torch.cat((torch.Tensor(real_position_1),torch.Tensor(real_position_2)),dim=0)
real_position=real_position.numpy()
real_position=real_position.reshape((Nnn)*heat_up_number,m+1)
y_mm=np.zeros((Nnn,m+1))
y_mm=torch.cat((torch.Tensor(y_mm_1),torch.Tensor(y_mm_2)),dim=0)                             
y_mm=y_mm.numpy()
real_position_new=np.zeros((Nnn*heat_up_number,m+1))


for i in range(0,Nnn*heat_up_number):
    real_position_new[i,0:897]=np.arange(0,897)
    real_position_new[i,897]=896.5
    real_position_new[i,898]=897
    real_position_new[i,899]=897.5
    real_position_new[i,900]=898
    real_position_new[i,0]=0

#interpolate
for i in range(0,Nnn*heat_up_number):
    f = interpolate.interp1d(real_position[i,:], XX[i,:])
    XX_new[i,:]=f(real_position_new[i,:]) 

XX_new=XX_new.reshape(Nnn,heat_up_number,m+1)

#normalize temp
for i in range(Nnn):
    XX_new[i,:,:]=(XX_new[i,:,:]-np.min(XX_new[i,:,:]))/(np.max(XX_new[i,:,:])-np.min(XX_new[i,:,:]))
#random choose injection/validation data set
N_validate=200
real_position_new=real_position_new.reshape(Nnn,heat_up_number,m+1)
Index=random.sample(range(0,Nnn),Nnn-N_validate)
X_train=np.zeros((Nnn-N_validate,heat_up_number,len(XX_new[0,0,:])))
y_train=np.zeros((Nnn-N_validate,len(XX_new[0,0,:])))
X_test=[]
y_test=[]
for j in range(0,Nnn-N_validate):
    X_train[j]=XX_new[Index[j],:,:]
    y_train[j]=y_mm[Index[j],:]   
aa=list(np.arange(0,Nnn))
cc=[x for x in aa if x in Index]
dd=[y for y in (aa+list(Index))if y not in cc]
for h in range(len(dd)):
    XX_new[dd[h],:,:]=XX_new[dd[h],:,:]
    X_test.append(XX_new[dd[h],:,:])
    y_test.append(y_mm[dd[h],:])


y_train_tensors = torch.LongTensor(y_train)
y_test_tensors = torch.LongTensor(np.array(y_test))

#add noise to input
noise_std_input = 0
X_train_noisy = X_train + noise_std_input * np.random.randn(*np.array(X_train).shape)
X_test_noisy=X_test + noise_std_input * np.random.randn(*np.array(X_test).shape)
X_train_tensors=torch.Tensor(X_train_noisy)
X_test_tensors=torch.Tensor(X_test_noisy)
class TemDataset(Dataset):
    def __init__(self,X_train_tensors,y_train_tensors):
        self.x=X_train_tensors
        self.y=y_train_tensors
        self.n_samples=self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples

dataset=TemDataset(X_train_tensors,y_train_tensors)
dataloader=DataLoader(dataset=dataset,batch_size=256,shuffle=True)#444
print("Training Shape", X_train_tensors.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors.shape, y_test_tensors.shape)

num_epochs = 800 #3000 epochs
learning_rate = 0.001#0.0005 #0.00001 lr
fcn = FCN().to(device) 
weight=torch.Tensor([1,70]).to(device)#2000
criterion = torch.nn.CrossEntropyLoss(weight=weight)#weighted crossentropy loss
optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate) 
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=50,
 verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
losses=[]
for epoch in range(num_epochs):
    for i,(inputs,labels) in enumerate(dataloader):  
        outputs = fcn.forward(inputs.to(device)) #forward pass
        loss = criterion(outputs, labels.to(device)) #forCrossEntropyLoss0411
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 
    if epoch % 100 == 0:
              losses.append(loss.item())
              print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))   
                


# In[ ]:


#algorithm
heat_up_number=1# number of input feacture
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,embed_size=40*2,heads=1,drop_prob=0):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.embed_size=embed_size
        self.heads=heads
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True,dropout=drop_prob) #lstm
        self.gru=nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=drop_prob)
        self.multiattention=nn.MultiheadAttention(embed_size,heads,dropout=drop_prob)
        self.ln=nn.LayerNorm(normalized_shape=seq_length,eps=1.0e-5)
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to(device) #internal state
        x=self.ln(x.transpose(2,1))#add layer_norm
        x=x.transpose(2,1)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = output
        return out    
class FCN(nn.Module):
    def __init__(self,c_in=heat_up_number, c_out=1, layers=[128, 256, 128], kss=[7, 5, 3],embed_size1=128,embed_size2=256,heads=1,drop_prob=0):#c_in是feacture number
        super(FCN, self).__init__()
        assert len(layers) == len(kss)
        self.convblock1 = nn.Conv1d(c_in, layers[0], kss[0])
        self.convblock2 = nn.Conv1d(layers[0], layers[1], kss[1])
        self.convblock3 = nn.Conv1d(layers[1], layers[2], kss[2])
        self.gap=torch.nn.AdaptiveAvgPool1d((m+1))
        self.fc = nn.Linear((layers[2]+2*40), 10)
        self.fcc = nn.Linear(10,2)
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
        self.prelu=nn.PReLU()
        self.tanh=nn.Tanh()
        self.bn1=nn.BatchNorm1d(layers[0],eps=1e-03,momentum=0.99)
        self.bn2=nn.BatchNorm1d(layers[1],eps=1e-03,momentum=0.99)
        self.bn3=nn.BatchNorm1d(layers[2],eps=1e-03,momentum=0.99)
        self.dropout=nn.Dropout(p=0)
        self.lstm1 = LSTM1(num_classes=m+1, input_size=heat_up_number, hidden_size=40, num_layers=2,seq_length=m+1).to(device)#input_size=feacture number
    def forward(self, x):
        x_fcn = self.convblock1(x)
        x_fcn=self.bn1(x_fcn)
        x_fcn=self.relu(x_fcn)
        x_fcn = self.convblock2(x_fcn)
        x_fcn=self.bn2(x_fcn)
        x_fcn=self.relu(x_fcn)
        x_fcn = self.convblock3(x_fcn)
        x_fcn=self.bn3(x_fcn)
        x_fcn=self.relu(x_fcn)
        x_fcn = self.gap(x_fcn)#size(N,C,m+1)
        x_fcn=x_fcn.transpose(1,2)
        ####LSTM
        x_lstm=x.transpose(1,2)
        x_lstm=self.lstm1(x_lstm)
        ###concat
        out_concat = torch.cat((x_lstm,x_fcn),2)
        out_concat=self.prelu(out_concat)
        out_concat=self.fc(out_concat)
        out_concat=self.prelu(out_concat)
        out_concat=self.fcc(out_concat)#size(N,m+1,2)
        outt=out_concat.transpose(1,2)#size(N,2,m+1)
        out_concat=self.prelu(outt)
        out_concat=self.softmax(outt)#calculate softmax along dim=1
        return out_concat

num_epochs = 800 #3000 epochs
learning_rate = 0.001#0.0005 #0.00001 lr
fcn = FCN().to(device) 
weight=torch.Tensor([1,70]).to(device)#2000
criterion = torch.nn.CrossEntropyLoss(weight=weight)#weighted crossentropy loss
optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate) 
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=50,
 verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
losses=[]
for epoch in range(num_epochs):
    for i,(inputs,labels) in enumerate(dataloader):  
        outputs = fcn.forward(inputs.to(device)) #forward pass
        loss = criterion(outputs, labels.to(device)) #forCrossEntropyLoss0411
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 
    if epoch % 100 == 0:
              losses.append(loss.item())
              print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))   


# In[ ]:


#accuracy matrix:
train_predict1 = model(X_test_tensors.to(device))#forward pass
train_predict1=train_predict1.cpu()
data_predict = train_predict1.data.numpy() #numpy conversion
dataY_plot = y_test_tensors.data.numpy()
data_predict1=data_predict[:,1,:]
label_predicted=np.argwhere(np.around(data_predict1)==1)
label_real1=np.argwhere(dataY_plot==1)
sum_real=0
sum_predicted=0
for i in range (label_real1.shape[0]):
    for any in data_predict1[label_real1[i][0],label_real1[i][1]-4:label_real1[i][1]+5]:
        if any>=0.5:
            sum_real=sum_real+1
            break
for j in range(label_predicted.shape[0]):
    for any in dataY_plot[label_predicted[j][0],label_predicted[j][1]-4:label_predicted[j][1]+5]:
        if any==1:
            sum_predicted=sum_predicted+1
            break
accuracy=2/(1/(sum_real/label_real1.shape[0])+1/(sum_predicted/label_predicted.shape[0]))               


# In[ ]:


#real case test
#load well-trained model
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset,DataLoader
import numpy as np
from scipy import interpolate
import random
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    #device = torch.device("cpu")
else:
    device = torch.device("cpu")

m=900
Nnn=800#
heat_up_number=1
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,embed_size=40*2,heads=1,drop_prob=0):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.embed_size=embed_size
        self.heads=heads
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,bidirectional=True,dropout=drop_prob) #lstm
        self.gru=nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,dropout=drop_prob)
        self.multiattention=nn.MultiheadAttention(embed_size,heads,dropout=drop_prob)
        self.ln=nn.LayerNorm(normalized_shape=seq_length,eps=1.0e-5)
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to(device) #internal state
        x=self.ln(x.transpose(2,1))#add layer_norm
        x=x.transpose(2,1)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = output
        return out    
class FCN(nn.Module):
    def __init__(self,c_in=heat_up_number, c_out=1, layers=[128, 256, 128], kss=[7, 5, 3],embed_size1=128,embed_size2=256,heads=1,drop_prob=0):#c_in是feacture number
        super(FCN, self).__init__()
        assert len(layers) == len(kss)
        self.convblock1 = nn.Conv1d(c_in, layers[0], kss[0])
        self.convblock2 = nn.Conv1d(layers[0], layers[1], kss[1])
        self.convblock3 = nn.Conv1d(layers[1], layers[2], kss[2])
        self.gap=torch.nn.AdaptiveAvgPool1d((m+1))
        self.fc = nn.Linear((layers[2]+2*40), 10)
        self.fcc = nn.Linear(10,2)
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
        self.prelu=nn.PReLU()
        self.tanh=nn.Tanh()
        self.bn1=nn.BatchNorm1d(layers[0],eps=1e-03,momentum=0.99)
        self.bn2=nn.BatchNorm1d(layers[1],eps=1e-03,momentum=0.99)
        self.bn3=nn.BatchNorm1d(layers[2],eps=1e-03,momentum=0.99)
        self.dropout=nn.Dropout(p=0)
        self.lstm1 = LSTM1(num_classes=m+1, input_size=heat_up_number, hidden_size=40, num_layers=2,seq_length=m+1).to(device)#input_size=feacture number
    def forward(self, x):
        x_fcn = self.convblock1(x)
        x_fcn=self.bn1(x_fcn)
        x_fcn=self.relu(x_fcn)
        x_fcn = self.convblock2(x_fcn)
        x_fcn=self.bn2(x_fcn)
        x_fcn=self.relu(x_fcn)
        x_fcn = self.convblock3(x_fcn)
        x_fcn=self.bn3(x_fcn)
        x_fcn=self.relu(x_fcn)
        x_fcn = self.gap(x_fcn)#size(N,C,m+1)
        x_fcn=x_fcn.transpose(1,2)
        ####LSTM
        x_lstm=x.transpose(1,2)
        x_lstm=self.lstm1(x_lstm)
        ###concat
        out_concat = torch.cat((x_lstm,x_fcn),2)
        out_concat=self.prelu(out_concat)
        out_concat=self.fc(out_concat)
        out_concat=self.prelu(out_concat)
        out_concat=self.fcc(out_concat)#size(N,m+1,2)
        outt=out_concat.transpose(1,2)#size(N,2,m+1)
        out_concat=self.prelu(outt)
        out_concat=self.softmax(outt)#calculate along dim=1
        return out_concat
    
model=torch.load('0315_new_single_new_type_fracture_prediction_99_67.pt')
os.chdir('/kaggle/input/ikeda11')
m=900
realposition=np.array(pd.read_csv('ikeda11.csv'))[0,:]
temp=np.array(pd.read_csv('ikeda11.csv'))[1:,:]
XX_new1=np.zeros((1,m+1))
real_position_new1=np.zeros((1,m+1))
real_position_new1[0,0:901]=np.arange(902.108,1585.24,683/900)[0:901]
f = interpolate.interp1d(realposition, temp[0,:])
XX_new1=(XX_new1-np.min(XX_new1))/(np.max(XX_new1)-np.min(XX_new1))
XX_new1=XX_new1.reshape((1,1,m+1))
X_test_tensors1= Variable(torch.Tensor(XX_new1))

#figure 7 in paper
fig = plt.figure()
ax = fig.add_subplot(111)
train_predict1 = model(X_test_tensors1.to(device))#forward pass
train_predict1=train_predict1.cpu()
data_predict = train_predict1.data.numpy() #numpy conversion
ax.plot(np.ones(11)*1115,np.arange(0,1.1,0.1),":",linewidth=2,color="#808080")    
ax.plot(np.ones(11)*1495,np.arange(0,1.1,0.1),":",linewidth=2,color="#808080") 
ax.plot(np.ones(11)*1540,np.arange(0,1.1,0.1),":",linewidth=2,color="#808080",label="Labeled fractures") 
ax.plot(real_position_new1[0,:],np.round(data_predict[0,1,:]),"r<",label="Predicted fractures")
ax2 = ax.twinx()
ax2.plot(real_position_new1[0,:],XX_new1[0,0,:],"b-",linewidth=2,label="Static temperature")
fig.legend(loc=6, bbox_to_anchor=(0.4,0.25))#, bbox_transform=ax.transAxes)
ax.set_ylabel("Fracture probability/[-]",color="r")
ax2.set_ylabel("Dimensionless wellbore temperature/[-]",color="b")
ax.set_xlabel("Depth/m")
plt.show()

