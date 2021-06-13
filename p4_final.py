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

bpsk = iqmodule.OpenIQ('IQ_4_bpsk_train.dat')
qpsk = iqmodule.OpenIQ('IQ_4_qpsk_train.dat')
qam16 = iqmodule.OpenIQ('IQ_4_qam16_train.dat')
psk8 = iqmodule.OpenIQ('IQ_4_psk8_train.dat')
print(bpsk.keys())

def to_slots(dat):
    #Split Dat into Timeslots
    i_s=[]
    q_s=[]
    for ts in range(int(dat.shape[1]/10000)):
        i_s.append(dat[0,10000*ts:10000*(ts+1)])
        q_s.append(dat[1,10000*ts:10000*(ts+1)])

    i_s=np.array(i_s)
    q_s=np.array(q_s)
    return i_s,q_s

#y_band, y_mod
total_i=[]
total_q=[]
for mod in [bpsk,qpsk,qam16,psk8]:
    mod_i=[]
    mod_q=[]
    for freq in range(1,10):
        i,q=to_slots(mod[freq])
        mod_i.append(i)
        mod_q.append(q)

    i_s=np.concatenate(mod_i,axis=0)
    q_s=np.concatenate(mod_q,axis=0)

    total_i.append(i_s)
    total_q.append(q_s)

I=np.concatenate(total_i,axis=0)
Q=np.concatenate(total_q,axis=0)
print(I.shape)
#y_mod
y_mod=[0]*180+[1]*180+[2]*180+[3]*180
y_mod=np.array(y_mod)

y_mod=y_mod.astype(np.int64)

y_band=[]
for idx in range(4):
    for j in range(9):
        y_band+=[j]*20
y_band=np.array(y_band)

y_band=y_band.astype(np.int64)

#fft
conv=[]
for idx in range(I.shape[0]):
    signal=I[idx]+Q[idx]*1j
    converted = np.fft.fft(signal)# / len(signal)  
    converted=abs(converted)
    conv.append(converted)
x_f=np.array(conv)

#Make Trainables
x_2=np.concatenate([np.expand_dims(I,axis=1),np.expand_dims(Q,axis=1)],axis=1)

x_3=np.concatenate([x_2,np.expand_dims(x_f,axis=1)],axis=1)
print(x_f.shape)
print(x_2.shape)
print(x_3.shape)

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
    def __init__(self, x,y_mod,y_band):
        self.x_data = torch.from_numpy(x) # size [n_samples, n_features]
        self.y_mod = torch.from_numpy(y_mod) # size [n_samples, 1]
        self.y_mod = self.y_mod.type(torch.LongTensor)

        self.y_band = torch.from_numpy(y_band) # size [n_samples, 1]
        self.y_band = self.y_band.type(torch.LongTensor)

    def __getitem__(self, index):
        return self.x_data[index], self.y_mod[index], self.y_band[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.x_data.shape[0]


x_train,x_val,y_mod_train,y_mod_val,y_band_train,y_band_val=train_test_split(x_3,y_mod,y_band, test_size=0.1, shuffle=True)

train_ds=IQDataset(x_train,y_mod_train,y_band_train)
train_dl = DataLoader(train_ds, batch_size=32)

val_ds=IQDataset(x_val,y_mod_val,y_band_val)
val_dl = DataLoader(val_ds, batch_size=32)
print(x_train.shape)
# exit()

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        # self.channel_num_in = 2000
        ch1=16
        ch2=16
        hid1=25*ch2
        hid2=64
        self.conv= nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=ch1,kernel_size=10,stride=10),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv1d(in_channels=ch1,out_channels=ch2,kernel_size=10,stride=10),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),

            # nn.Conv1d(in_channels=ch2,out_channels=ch3,kernel_size=2,stride=2),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Flatten()
        )
        self.clf_mod=nn.Sequential(
            nn.Linear(hid1, hid2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),

            # nn.Linear(hid2, hid3),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(hid2, 4),
            nn.Softmax(dim=1)
        )

        self.clf_band=nn.Sequential(
            nn.Linear(hid1, hid2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),

            # nn.Linear(hid2, hid3),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(hid2, 9),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x=x.float()
        x=self.conv(x)
        # print(x.shape)
        # x.
        pred_mod = self.clf_mod(x)
        pred_band = self.clf_band(x)
        # print(pred.shape)
        return pred_mod,pred_band


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model=SimpleConv().to(device)
loss_fn = torch.nn.CrossEntropyLoss()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
model.apply(init_weights)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.zero_grad()
model.train(True)

#Train Model
train_model=False
if train_model:
    num_ep=250
    for ep in range(num_ep):
        model.train()
        for x_b,y_mod_b,y_band_b in train_dl:
            x_b=x_b.to(device)
            y_mod_b=y_mod_b.to(device)
            y_band_b=y_band_b.to(device)

            pred_mod,pred_band = model(x_b)
            y_mod_b=y_mod_b.type(torch.LongTensor).to(device)
            y_band_b=y_band_b.type(torch.LongTensor).to(device)

            loss_mod = loss_fn(pred_mod, y_mod_b)
            loss_band = loss_fn(pred_band, y_band_b)
            
            loss=loss_mod + loss_band

            loss.backward()
            opt.step()
            opt.zero_grad()
        
        #Val Acc
        if (ep+1)%1==0:
            print("Epoch {}".format(ep+1))
            model.eval()
            preds_mod=[]
            preds_band=[]
            val_losses=[]
            with torch.no_grad():
                for x_b,y_mod_b,y_band_b in train_dl:
                    x_b=x_b.to(device)
                    y_mod_b=y_mod_b.to(device)
                    y_band_b=y_band_b.to(device)

                    pred_mod,pred_band = model(x_b)
                    y_mod_b=y_mod_b.type(torch.LongTensor).to(device)
                    y_band_b=y_band_b.type(torch.LongTensor).to(device)

                    loss_mod = loss_fn(pred_mod, y_mod_b)
                    loss_band = loss_fn(pred_band, y_band_b)
                    
                    loss=0.5*loss_mod + 0.5*loss_band

                    val_losses.append(loss.item())

                    preds_mod.append(np.argmax(pred_mod.cpu().detach().numpy(),axis=1))
                    preds_band.append(np.argmax(pred_band.cpu().detach().numpy(),axis=1))

                y_pred_mod=np.concatenate(preds_mod,axis=0)
                y_pred_band=np.concatenate(preds_band,axis=0)

                print("Train Loss {:.5f}".format(sum(val_losses)/len(val_losses)))
            train_loss=sum(val_losses)/len(val_losses)
            accuracy_mod = accuracy_score(y_mod_train,y_pred_mod)
            accuracy_band = accuracy_score(y_band_train,y_pred_band)

            print("Accuracy Mod: {:0.4f}".format(accuracy_mod))
            print("Accuracy Band: {:0.4f}".format(accuracy_band))
            print(classification_report(y_mod_train, y_pred_mod, target_names=['bpsk','qpsk','qam16','psk8']))
            print(classification_report(y_band_train, y_pred_band, target_names=['1','2','3','4','5','6','7','8','9']))

            preds_mod=[]
            preds_band=[]
            val_losses=[]
            with torch.no_grad():
                for x_b,y_mod_b,y_band_b in val_dl:
                    x_b=x_b.to(device)
                    y_mod_b=y_mod_b.to(device)
                    y_band_b=y_band_b.to(device)

                    pred_mod,pred_band = model(x_b)
                    y_mod_b=y_mod_b.type(torch.LongTensor).to(device)
                    y_band_b=y_band_b.type(torch.LongTensor).to(device)

                    loss_mod = loss_fn(pred_mod, y_mod_b)
                    loss_band = loss_fn(pred_band, y_band_b)
                    
                    loss=0.5*loss_mod + 0.5*loss_band

                    val_losses.append(loss.item())

                    preds_mod.append(np.argmax(pred_mod.cpu().detach().numpy(),axis=1))
                    preds_band.append(np.argmax(pred_band.cpu().detach().numpy(),axis=1))

                y_pred_mod=np.concatenate(preds_mod,axis=0)
                y_pred_band=np.concatenate(preds_band,axis=0)

                print("Val Loss {:.5f}".format(sum(val_losses)/len(val_losses)))
            val_loss=sum(val_losses)/len(val_losses)

            accuracy_mod = accuracy_score(y_mod_val,y_pred_mod)
            accuracy_band = accuracy_score(y_band_val,y_pred_band)

            print("Accuracy Mod: {:0.4f}".format(accuracy_mod))
            print("Accuracy Band: {:0.4f}".format(accuracy_band))
            print(classification_report(y_mod_val, y_pred_mod, target_names=['bpsk','qpsk','qam16','psk8']))
            print(classification_report(y_band_val, y_pred_band, target_names=['1','2','3','4','5','6','7','8','9']))

    #Final
    print("Train Complete")
    #Save Model
    torch.save(model.state_dict(), 'p4.pt')

model.load_state_dict(torch.load('p4.pt'))

model.eval()
preds_mod=[]
preds_band=[]
val_losses=[]
with torch.no_grad():
    for x_b,y_mod_b,y_band_b in val_dl:
        x_b=x_b.to(device)
        y_mod_b=y_mod_b.to(device)
        y_band_b=y_band_b.to(device)

        pred_mod,pred_band = model(x_b)
        y_mod_b=y_mod_b.type(torch.LongTensor).to(device)
        y_band_b=y_band_b.type(torch.LongTensor).to(device)

        loss_mod = loss_fn(pred_mod, y_mod_b)
        loss_band = loss_fn(pred_band, y_band_b)
        
        loss=0.5*loss_mod + 0.5*loss_band

        val_losses.append(loss.item())

        preds_mod.append(np.argmax(pred_mod.cpu().detach().numpy(),axis=1))
        preds_band.append(np.argmax(pred_band.cpu().detach().numpy(),axis=1))

    y_pred_mod=np.concatenate(preds_mod,axis=0)
    y_pred_band=np.concatenate(preds_band,axis=0)

    print("Val Loss {:.5f}".format(sum(val_losses)/len(val_losses)))
val_loss=sum(val_losses)/len(val_losses)

accuracy_mod = accuracy_score(y_mod_val,y_pred_mod)
accuracy_band = accuracy_score(y_band_val,y_pred_band)

print("Accuracy Mod: {:0.4f}".format(accuracy_mod))
print("Accuracy Band: {:0.4f}".format(accuracy_band))
print(classification_report(y_mod_val, y_pred_mod, target_names=['bpsk','qpsk','qam16','psk8']))
print(classification_report(y_band_val, y_pred_band, target_names=['1','2','3','4','5','6','7','8','9']))


#########################################################################
# < Receive IQ data >
# Put your team id and problem number for receiving IQ data
# If the server is off, you will see a message below
# code32: Server is not working
#########################################################################
# import uclient
# TeamID = 5708          # enter team id 
# ProbNumber = 4      # enter problem number

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
# Y_mod=np.zeros(X.shape[0])
# Y_band=np.zeros(X.shape[0])

# test_ds=IQDataset(X,Y_mod,Y_band)
# test_dl = DataLoader(test_ds, batch_size=32)

# model.eval()
# preds_mod=[]
# preds_band=[]
# with torch.no_grad():
#     for x_b,y_mod_b,y_band_b in test_dl:
#         x_b=x_b.to(device)
#         y_mod_b=y_mod_b.to(device)
#         y_band_b=y_band_b.to(device)

#         pred_mod,pred_band = model(x_b)
#         preds_mod.append(np.argmax(pred_mod.cpu().detach().numpy(),axis=1))
#         preds_band.append(np.argmax(pred_band.cpu().detach().numpy(),axis=1))

#     y_pred_mod=np.concatenate(preds_mod,axis=0)
#     y_pred_band=np.concatenate(preds_band,axis=0)

# AnsVector = []
# for idx in range(IQdata.shape[0]):
#     AnsVector.append([y_pred_band[idx],y_pred_mod[idx]])
# print(AnsVector[:5])
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
