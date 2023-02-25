#!/usr/bin/python3
"""Trains a Recurrent neural network, taking a series a numbers as input and outputing the sum in the end.
The input numbers are the images from the MNIST dataset. The length of the sequence of numbers is random between 1 and 15.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms

## Class definition
class Net(nn.Module):
    """A network computing the sum of a sequence of N numbers. The numbers are given as a sequence of MNIST images.
    The network outputs a sequence of size N+2: the N first are the numbers read as input, the 2 last are the decimal and the unit of the sum of the sequence of numbers.
    
    The network is a one layer Recurrent Neural Network (RNN), followed by a fully connected layer. The cell of the RNN is a Gated Recurrent Unit (GRU)."""
    def __init__(self):
        super(Net, self).__init__()
        self.inpt = nn.Linear(28*28, 100)
        self.rtnn = nn.GRU(100, 10, batch_first=True)
        self.oupt = nn.Linear(10, 10)
    
    def forward(self, x):
        oupt, hddn = self.rtnn(self.inpt(x))
        return self.oupt(oupt), hddn

class DataAddition(torch.utils.data.Dataset):
    def __init__(self, data, trgt):
        super(DataAddition, self).__init__()
        self.data = data
        self.trgt = trgt
    
    def __getitem__(self, index):
        return (self.data[index], self.trgt[index])
    
    def __len__ (self):
        return len(self.data)

## Data pre-processing
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
id_mnist = 0
sequences = []
targets = []
rmng = mnist_train.data.shape[0]
while rmng > 0:
    lgth = torch.randint(1, 11, (1,))[0]
    if rmng - lgth < 0:
        lgth = rmng
    
    rmng -= lgth
    sequences.append(torch.concat((mnist_train.data[id_mnist:id_mnist+lgth].flatten(1), torch.zeros((1, 28*28)), torch.zeros((1, 28*28)))))
    s = mnist_train.targets[id_mnist:id_mnist+lgth].sum()
    targets.append(torch.concat((mnist_train.targets[id_mnist:id_mnist+lgth], (s//10).unsqueeze(0), (s - 10*(s//10)).unsqueeze(0))))
    id_mnist += lgth

train_loader = torch.utils.data.DataLoader(DataAddition(nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1), nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)), batch_size=32)

mnist_eval = datasets.MNIST('../data', train=False, transform=transform)
id_mnist = 0
sequences = []
targets = []
rmng = mnist_eval.data.shape[0]
while rmng > 0:
    lgth = torch.randint(1, 11, (1,))[0]
    if rmng - lgth < 0:
        lgth = rmng
    
    rmng -= lgth
    sequences.append(torch.concat((mnist_eval.data[id_mnist:id_mnist+lgth].flatten(1), torch.zeros((1, 28*28)), torch.zeros((1, 28*28)))))
    s = mnist_eval.targets[id_mnist:id_mnist+lgth].sum()
    targets.append(torch.concat((mnist_eval.targets[id_mnist:id_mnist+lgth], (s//10).unsqueeze(0), (s - 10*(s//10)).unsqueeze(0))))
    id_mnist += lgth

eval_loader = torch.utils.data.DataLoader(DataAddition(nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1), nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)), batch_size=mnist_eval.data.shape[0])

## Parameters and hyper-parameters
MAXM_BTCH_SIZE = 32
MAXM_EPCH = 100
LRNG_RATE = 1e-2
STEP_SIZE = 1
TIME_ZERO = 10
TIME_MULT = 2
GMMA = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fnct = nn.CrossEntropyLoss(reduction="none", ignore_index=-1).to(device)

################################################################################
##                                   LOSS 1                                   ##
################################################################################
torch.manual_seed(123456789)

modl1 = Net().to(device)
optm1 = optim.Adadelta(modl1.parameters(), lr=LRNG_RATE)

schd1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optm1, T_0=TIME_ZERO, T_mult=TIME_MULT)
modl1.train()
epch_loss1 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm1.zero_grad()
        oupt, _ = modl1(data)
        loss = 0
        btch_size = trgt.shape[0]
        for idx in range(trgt.shape[1]):
            tmp = loss_fnct(oupt[:, idx, :], trgt[:, idx])
            epch_loss1[epch, 0:btch_size] = torch.from_numpy(np.array([tmp[i].item() for i in range(len(tmp))]))
            epch_loss1[epch, MAXM_BTCH_SIZE] = tmp.sum().item()
            loss += loss_fnct(oupt[:, idx, :], trgt[:, idx]).sum()
        
        loss.backward()
        optm1.step()
    
    print("Epoch {} | loss {}".format(epch, loss), flush=True)
    schd1.step()

################################################################################
#                                   LOSS 2                                   ##
################################################################################
torch.manual_seed(123456789)

modl2 = Net().to(device)
optm2 = optim.Adadelta(modl2.parameters(), lr=LRNG_RATE)

schd2 = StepLR(optm2, step_size=STEP_SIZE, gamma=GMMA)
modl2.train()
epch_loss2 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm2.zero_grad()
        oupt, _ = modl2(data)
        
        ## Loss computation
        oupt = torch.permute(oupt, (0, 2, 1))
        btch_size = trgt.shape[0]
        tmp = loss_fnct(oupt, trgt)
        epch_loss2[epch, 0:btch_size] = torch.from_numpy(np.array([tmp[i].sum().item() for i in range(len(tmp))]))
        epch_loss2[epch, MAXM_BTCH_SIZE] = tmp.sum().item()
        
        loss = loss_fnct(oupt, trgt).sum()
        loss.backward()
        optm2.step()
    
    print("Epoch {} | loss {}".format(epch, loss), flush=True)
    schd2.step()

################################################################################
##                                   LOSS 3                                   ##
################################################################################
torch.manual_seed(123456789)

modl3 = Net().to(device)
optm3 = optim.Adadelta(modl3.parameters(), lr=LRNG_RATE)

schd3 = StepLR(optm3, step_size=STEP_SIZE, gamma=GMMA)
modl3.train()
epch_loss3 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm3.zero_grad()
        oupt, _ = modl3(data)
        
        ## Loss computation
        oupt = oupt.swapaxes(2, 1)
        btch_size = trgt.shape[0]
        tmp = loss_fnct(oupt, trgt)
        epch_loss3[epch, 0:btch_size] = torch.from_numpy(np.array([tmp[i].sum().item() for i in range(len(tmp))]))
        epch_loss3[epch, MAXM_BTCH_SIZE] = tmp.sum().item()
        
        loss = loss_fnct(oupt, trgt).sum()
        loss.backward()
        optm3.step()
    
    print("Epoch {} | loss {}".format(epch, loss), flush=True)
    schd3.step()

################################################################################
##                                   LOSS 4                                   ##
################################################################################
torch.manual_seed(123456789)

modl4 = Net().to(device)
optm4 = optim.Adadelta(modl4.parameters(), lr=LRNG_RATE)

schd4 = StepLR(optm4, step_size=STEP_SIZE, gamma=GMMA)
modl4.train()
epch_loss4 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm4.zero_grad()
        oupt, _ = modl4(data)
        
        ## Loss computation
        oupt = oupt.swapdims(2, 1)
        btch_size = trgt.shape[0]
        tmp = loss_fnct(oupt, trgt)
        epch_loss4[epch, 0:btch_size] = torch.from_numpy(np.array([tmp[i].sum().item() for i in range(len(tmp))]))
        epch_loss4[epch, MAXM_BTCH_SIZE] = tmp.sum().item()
        
        loss = loss_fnct(oupt, trgt).sum()
        loss.backward()
        optm4.step()
    
    print("Epoch {} | loss {}".format(epch, loss), flush=True)
    schd4.step()

################################################################################
##                            EVALUATE PERFORMANCE                            ##
################################################################################
from sklearn.metrics import roc_auc_score
_, (data, trgt) = next(enumerate(eval_loader))

modl1.eval()
oupt, _ = modl1(data)
print("Area Under the Curve of model {}\n{}".format(1, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl2.eval()
oupt, _ = modl2(data)
print("Area Under the Curve of model {}\n{}".format(2, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl3.eval()
oupt, _ = modl3(data)
print("Area Under the Curve of model {}\n{}".format(3, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl4.eval()
oupt, _ = modl4(data)
print("Area Under the Curve of model {}\n{}".format(4, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))


################################################################################
##                              COMPARE NETWORKS                              ##
################################################################################
def compare_models(model1, model2, precision):
    return {"Input (Linear)": \
            ((model1.inpt.weight[0][0] >= model2.inpt.weight[0][0] - precision).all() and (model1.inpt.weight[0][0] <= model2.inpt.weight[0][0] + precision).all(),
            (model1.inpt.bias >= model2.inpt.bias - precision).all() and (model1.inpt.bias <= model2.inpt.bias + precision).all()),
            "Core (GRU)": \
            ((model1.rtnn.all_weights[0][0] >= model2.rtnn.all_weights[0][0] - precision).all() and (model1.rtnn.all_weights[0][0] <= model2.rtnn.all_weights[0][0] + precision).all(),
            (model1.rtnn.all_weights[0][1] >= model2.rtnn.all_weights[0][1] - precision).all() and (model1.rtnn.all_weights[0][1] <= model2.rtnn.all_weights[0][1] + precision).all(),
            (model1.rtnn.all_weights[0][2] >= model2.rtnn.all_weights[0][2] - precision).all() and (model1.rtnn.all_weights[0][2] <= model2.rtnn.all_weights[0][2] + precision).all(),
            (model1.rtnn.all_weights[0][3] >= model2.rtnn.all_weights[0][3] - precision).all() and (model1.rtnn.all_weights[0][3] <= model2.rtnn.all_weights[0][3] + precision).all()),
            "Output (Linear)": \
            ((model1.oupt.weight[0][0] >= model2.oupt.weight[0][0] - precision).all() and (model1.oupt.weight[0][0] <= model2.oupt.weight[0][0] + precision).all(),
            (model1.oupt.bias >= model2.oupt.bias - precision).all() and (model1.oupt.bias <= model2.oupt.bias + precision).all()),
           }

precision = 1e-10
print("Compare the state of model {}/{} (precision = {}): {}".format(1, 2, precision, compare_models(modl1, modl2, precision)))
print("Compare the state of model {}/{} (precision = {}): {}".format(2, 3, precision, compare_models(modl1, modl3, precision)))
print("Compare the state of model {}/{} (precision = {}): {}".format(1, 4, precision, compare_models(modl1, modl4, precision)))

print("Compare the state of model {}/{} (precision = {}): {}".format(1, 3, precision, compare_models(modl2, modl3, precision)))
print("Compare the state of model {}/{} (precision = {}): {}".format(1, 4, precision, compare_models(modl2, modl4, precision)))

print("Compare the state of model {}/{} (precision = {}): {}".format(3, 4, precision, compare_models(modl3, modl4, precision)))
