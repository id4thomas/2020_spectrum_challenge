#########################################################################
# Write your algorithm below
#########################################################################
import iqmodule

import numpy as np
import random

from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


seed_num=42
random.seed(seed_num)
np.random.seed(seed_num)

iq=iqmodule.OpenIQ('IQ_3_train.dat')
bpsk = iq['bpsk']
qpsk = iq['qpsk']
qam16 = iq['qam16']
psk8=iq['psk8']
print(bpsk.shape)

def to_slots(dat):
    #Split Dat into Timeslots
    i_s=[]
    q_s=[]
    for ts in range(int(dat.shape[1]/1000)):
        i_s.append(dat[0,1000*ts:1000*(ts+1)])
        q_s.append(dat[1,1000*ts:1000*(ts+1)])

    i_s=np.array(i_s)
    q_s=np.array(q_s)
    return i_s,q_s

i_bpsk,q_bpsk=to_slots(bpsk)
i_qpsk,q_qpsk=to_slots(qpsk)
i_qam16,q_qam16=to_slots(qam16)
i_psk8,q_psk8=to_slots(psk8)

i=np.concatenate([i_bpsk,i_qpsk,i_qam16,i_psk8],axis=0)
q=np.concatenate([q_bpsk,q_qpsk,q_qam16,q_psk8],axis=0)
print(i.shape)
x=np.concatenate([i,q],axis=1)
print(x.shape)

#2 Channel
i_2=np.expand_dims(i,axis=1)
q_2=np.expand_dims(q,axis=1)
x_2=np.concatenate([i_2,q_2],axis=1)
print(x_2.shape)

#fft conversion
conv=[]
for idx in range(i.shape[0]):
    signal=i[idx]+q[idx]*1j
    converted = np.fft.fft(signal)# / len(signal)  
    converted=abs(converted)
    conv.append(converted)
x_f=np.array(conv)
print(x_f.shape)

#Make labels
y_1=np.zeros(1000)
y_2=np.ones(1000)
y_3=np.ones(1000)*2
y_4=np.ones(1000)*3
y=np.concatenate([y_1,y_2,y_3,y_4],axis=0)
y=y.astype(np.int64)

#####DL Model
import torch
import torch.nn as nn
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class IQDataset(Dataset):
    def __init__(self, x,y):
        self.x_data = torch.from_numpy(x) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(y) # size [n_samples, 1]
        self.y_data = self.y_data.type(torch.LongTensor)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.x_data.shape[0]


#Split Train Val
def threshold(data, th):
    th = np.quantile(abs(data), th)
    for idx, item in enumerate(data):
        if item < th:
            data[idx] = 0
    return data, th

#Thresholded Data
# temp_fft = x_f.copy()
# for i in range(temp_fft.shape[0]):
#     temp_fft[i] = threshold(temp_fft[i], 0.05)[0]
# x_f_t = abs(temp_fft)
# x_f_t=np.expand_dims(x_f_t,axis=1)

x_f=np.expand_dims(x_f,axis=1)
# x_f=x_f.astype(np.float32)

x_3=np.concatenate([x_2,x_f],axis=1)
x_train,x_val,y_train,y_val=train_test_split(x_3,y, test_size=0.1, shuffle=True)

train_ds=IQDataset(x_train,y_train)
train_dl = DataLoader(train_ds, batch_size=32)

val_ds=IQDataset(x_val,y_val)
val_dl = DataLoader(val_ds, batch_size=32)
print(x_train.shape)
# exit()

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        # self.channel_num_in = 2000
        ch1=16
        ch2=16
        # ch3=128
        hid1=10*ch2
        hid2=64
        hid3=32
        self.conv= nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=ch1,kernel_size=5,stride=5),
            # nn.BatchNorm1d(ch1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5,stride=5),

            nn.Conv1d(in_channels=ch1,out_channels=ch2,kernel_size=2,stride=2),
            # nn.BatchNorm1d(ch2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),

            # nn.Conv1d(in_channels=ch2,out_channels=ch3,kernel_size=2,stride=2),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Flatten()
        )
        self.clf=nn.Sequential(
            nn.Linear(hid1, hid2),
            # nn.BatchNorm1d(hid2),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hid2, hid3),
            # nn.BatchNorm1d(hid3),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hid3, 4),
            nn.Softmax(dim=1))


    def forward(self, x):
        x=x.float()
        x=self.conv(x)
        # print(x.shape)
        # x.
        pred = self.clf(x)
        # print(pred.shape)
        return pred


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=SimpleConv().to(device)
loss_fn = torch.nn.CrossEntropyLoss()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
model.apply(init_weights)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.zero_grad()
model.train(True)

model_train=False
if model_train:
    num_ep=240
    for ep in range(num_ep):
        model.train()
        for x_b,y_b in train_dl:
            x_b=x_b.to(device)
            y_b=y_b.to(device)
            pred = model(x_b)
            y_b=y_b.type(torch.LongTensor).to(device)
            loss = loss_fn(pred, y_b)

            loss.backward()
            opt.step()
            opt.zero_grad()
        
        #Val Acc
        if (ep+1)%1==0:
            print("Epoch {}".format(ep+1))
            model.eval()
            preds=[]
            val_losses=[]
            with torch.no_grad():
                for x_b,y_b in train_dl:
                    x_b=x_b.to(device)
                    y_b=y_b.to(device)
                    pred = model(x_b)
                    y_b=y_b.type(torch.LongTensor).to(device)
                    loss = loss_fn(pred, y_b)
                    val_losses.append(loss.item())
                    preds.append(np.argmax(pred.cpu().detach().numpy(),axis=1))
                y_pred=np.concatenate(preds,axis=0)
                print("Train Loss {:.5f}".format(sum(val_losses)/len(val_losses)))
            train_loss=sum(val_losses)/len(val_losses)
            accuracy = accuracy_score(y_train,y_pred)
            print("Accuracy : {:0.4f}".format(accuracy))
            print(classification_report(y_train, y_pred, target_names=['bpsk','qpsk','qam16','psk8']))

            preds=[]
            val_losses=[]
            with torch.no_grad():
                for x_b,y_b in val_dl:
                    x_b=x_b.to(device)
                    y_b=y_b.to(device)
                    pred = model(x_b)
                    y_b=y_b.type(torch.LongTensor).to(device)
                    loss = loss_fn(pred, y_b)
                    val_losses.append(loss.item())
                    preds.append(np.argmax(pred.cpu().detach().numpy(),axis=1))
                y_pred=np.concatenate(preds,axis=0)
                print("Val Loss {:.5f}".format(sum(val_losses)/len(val_losses)))
            val_loss=sum(val_losses)/len(val_losses)
            accuracy = accuracy_score(y_val,y_pred)
            print("Accuracy : {:0.4f}".format(accuracy))
            print(classification_report(y_val, y_pred, target_names=['bpsk','qpsk','qam16','psk8']))

    #Final
    print("Train Complete")
    #Save Model
    torch.save(model.state_dict(), 'p3.pt')


model.load_state_dict(torch.load('p3.pt'))
model.eval()

preds=[]
val_losses=[]
with torch.no_grad():
    for x_b,y_b in val_dl:
        x_b=x_b.to(device)
        y_b=y_b.to(device)
        pred = model(x_b)
        y_b=y_b.type(torch.LongTensor).to(device)
        loss = loss_fn(pred, y_b)
        val_losses.append(loss.item())
        preds.append(np.argmax(pred.cpu().detach().numpy(),axis=1))
    y_pred=np.concatenate(preds,axis=0)
    print("Val Loss {:.5f}".format(sum(val_losses)/len(val_losses)))
accuracy = accuracy_score(y_val,y_pred)
print("Accuracy : {:0.4f}".format(accuracy))
print(classification_report(y_val, y_pred, target_names=['bpsk','qpsk','qam16','psk8']))


#########################################################################
# < Receive IQ data >
# Put your team id and problem number for receiving IQ data
# If the server is off, you will see a message below
# code32: Server is not working
#########################################################################
# import uclient
# TeamID =  5708         # enter team id 
# ProbNumber = 3      # enter problem number

# IQdata = uclient.ReceiveIQ(TeamID, ProbNumber)
# conv=[]
# for idx in range(IQdata.shape[0]):
#     signal=IQdata[idx,0]+IQdata[idx,1]*1j
#     converted = np.fft.fft(signal)# / len(signal)  
#     converted=abs(converted)
#     conv.append(converted)
# IQ_f=np.array(conv)
# print(IQ_f.shape)
# #Predict
# X=np.concatenate([IQdata,IQ_f],axis=1)
# print(X.shape)
# Y=np.zeros(X.shape[0])

# test_ds=IQDataset(X,Y)
# test_dl = DataLoader(test_ds, batch_size=32)

# preds=[]
# val_losses=[]
# with torch.no_grad():
#     for x_b,y_b in test_dl:
#         x_b=x_b.to(device)
#         y_b=y_b.to(device)
#         pred = model(x_b)
#         preds.append(np.argmax(pred.cpu().detach().numpy(),axis=1))

#     AnsVector=np.concatenate(preds,axis=0)
# print(AnsVector)
# AnsVector=AnsVector.tolist()
# print(len(AnsVector))

#########################################################################
# < Send Answer >
# Put your answer to AnsVector and send answer to the server
# please check the terminal message to see if the answer is sent
#
# If the Ansvector is not a proper varialble type or size of the Ansvector 
# is not correct, terminal will print a message below
# >> Check Answer data type and length please!
#
# If you send your answer correctly you will see a message below
# >> code10: Transmit success
#########################################################################

# uclient.SendResult(TeamID, ProbNumber, AnsVector)
