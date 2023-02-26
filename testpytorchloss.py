#!/usr/bin/python3
"""Trains a Recurrent neural network, taking a series a numbers as input and outputing the sum in the end.
The input numbers are the images from the MNIST dataset. The length of the sequence of numbers is random between 1 and 15.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms

import pandas as pd
log = open("log_results.txt", "w")

## Class definition
class Net(nn.Module):
    """A network computing the sum of a sequence of N numbers. The numbers are given as a sequence of MNIST images.
    The network outputs a sequence of size N+2: the N first are the numbers read as input, the 2 last are the decimal and the unit of the sum of the sequence of numbers.
    
    The network is a one layer Recurrent Neural Network (RNN), preceded and followed by a fully connected layer for embedding. The cell of the RNN is a Gated Recurrent Unit (GRU)."""
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

## Training data
mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
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

## Evaluation data
mnist_eval = datasets.MNIST('data', train=False, transform=transform)
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
TIME_ZERO = 10
TIME_MULT = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fnct = nn.CrossEntropyLoss(reduction="none", ignore_index=-1).to(device)
loss_fnct5 = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1).to(device)
loss_fnct6 = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1).to(device)

################################################################################
##                                   LOSS 1                                   ##
################################################################################
torch.manual_seed(123456789) # Fix the PRNG seed

# Network, optimizer, scheduler
modl1 = Net().to(device)
optm1 = optim.Adadelta(modl1.parameters(), lr=LRNG_RATE)
schd1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optm1, T_0=TIME_ZERO, T_mult=TIME_MULT)

modl1.train()
epch_loss1 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1)) # Store loss each epoch for later comparison
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm1.zero_grad()
        oupt, _ = modl1(data)
        
        loss = 0
        btch_size = trgt.shape[0]
        for idx in range(trgt.shape[1]):
            tmp = loss_fnct(oupt[:, idx, :], trgt[:, idx])
            epch_loss1[epch, 0:btch_size] += torch.from_numpy(np.array([tmp[i].item() for i in range(len(tmp))]))
            epch_loss1[epch, MAXM_BTCH_SIZE] += tmp.sum().item()
            loss += loss_fnct(oupt[:, idx, :], trgt[:, idx]).sum()
        
        loss.backward()
        optm1.step()
    
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
    
    schd4.step()

################################################################################
##                                   LOSS 5                                   ##
################################################################################
torch.manual_seed(123456789)

modl5 = Net().to(device)
optm5 = optim.Adadelta(modl5.parameters(), lr=LRNG_RATE)

schd5 = StepLR(optm5, step_size=STEP_SIZE, gamma=GMMA)
modl5.train()
epch_loss5 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm5.zero_grad()
        oupt, _ = modl5(data)
        
        ## Loss computation
        oupt = oupt.swapdims(2, 1)
        btch_size = trgt.shape[0]
        tmp = loss_fnct5(oupt, trgt)
        epch_loss5[epch, MAXM_BTCH_SIZE] = tmp.item()
        
        loss = loss_fnct5(oupt, trgt)
        loss.backward()
        optm5.step()
    
    schd5.step()

################################################################################
##                                   LOSS 6                                   ##
################################################################################
torch.manual_seed(123456789)

modl6 = Net().to(device)
optm6 = optim.Adadelta(modl6.parameters(), lr=LRNG_RATE)

schd6 = StepLR(optm6, step_size=STEP_SIZE, gamma=GMMA)
modl6.train()
epch_loss6 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm6.zero_grad()
        oupt, _ = modl6(data)
        
        ## Loss computation
        oupt = oupt.swapdims(2, 1)
        btch_size = trgt.shape[0]
        tmp = loss_fnct6(oupt, trgt)
        epch_loss6[epch, MAXM_BTCH_SIZE] = tmp.item()
        
        loss = loss_fnct6(oupt, trgt)
        loss.backward()
        optm6.step()
    
    schd6.step()

################################################################################
##                                   LOSS 7                                   ##
################################################################################
torch.manual_seed(123456789)

modl7 = Net().to(device)
optm7 = optim.Adadelta(modl7.parameters(), lr=LRNG_RATE)

schd7 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optm7, T_0=TIME_ZERO, T_mult=TIME_MULT)
modl7.train()
epch_loss7 = torch.zeros((MAXM_EPCH + 1, MAXM_BTCH_SIZE+1))
for epch in range(1, MAXM_EPCH + 1):
    for batch_idx, (data, trgt) in enumerate(train_loader):
        data, trgt = data.to(device), trgt.to(device)
        optm7.zero_grad()
        oupt, _ = modl7(data)
        loss = 0
        btch_size = trgt.shape[0]
        for idx in range(trgt.shape[1]):
            tmp = loss_fnct2(oupt[:, idx, :], trgt[:, idx])
            epch_loss7[epch, MAXM_BTCH_SIZE] = tmp.item()
            loss += loss_fnct2(oupt[:, idx, :], trgt[:, idx])
        
        loss.backward()
        optm7.step()
    
    schd7.step()


losses = [epch_loss1, epch_loss2, epch_loss3, epch_loss4, epch_loss5, epch_loss6, epch_loss7]
log = open("compare_losses.txt", "w")
idl = 1
for idl, l in enumerate(losses):
    log.write("# Loss " + str(idl) + "\n")
    idl += 1
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            log.write(str(l[i, j].item()) + ", ")
        
        log.write("\n")
    
    log.write("\n\n")

log.close()

################################################################################
##                            EVALUATE PERFORMANCE                            ##
################################################################################
from sklearn.metrics import roc_auc_score
_, (data, trgt) = next(enumerate(eval_loader))
log = open("compare_auroc.txt", "w")

modl1.eval()
oupt, _ = modl1(data)
log.write("Area Under the Curve of model {} | {}\n".format(1, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl2.eval()
oupt, _ = modl2(data)
log.write("Area Under the Curve of model {} | {}\n".format(2, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl3.eval()
oupt, _ = modl3(data)
log.write("Area Under the Curve of model {} | {}\n".format(3, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl4.eval()
oupt, _ = modl4(data)
log.write("Area Under the Curve of model {} | {}\n".format(4, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl5.eval()
oupt, _ = modl5(data)
log.write("Area Under the Curve of model {} | {}\n".format(5, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl6.eval()
oupt, _ = modl6(data)
log.write("Area Under the Curve of model {} | {}\n".format(6, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))

modl7.eval()
oupt, _ = modl7(data)
log.write("Area Under the Curve of model {} | {}\n".format(7, roc_auc_score(nn.functional.one_hot(trgt[torch.where(trgt!=-1)]).detach().numpy(), torch.softmax(oupt[torch.where(trgt!=-1)], 1).detach().numpy(), average=None)))
log.close()

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
log = open("compare_state.txt", "w")
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(1, 2, precision, compare_models(modl1, modl2, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(1, 3, precision, compare_models(modl1, modl3, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(1, 4, precision, compare_models(modl1, modl4, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(1, 5, precision, compare_models(modl1, modl5, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(1, 6, precision, compare_models(modl1, modl6, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(1, 7, precision, compare_models(modl1, modl7, precision)))

log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(2, 3, precision, compare_models(modl2, modl3, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(2, 4, precision, compare_models(modl2, modl4, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(2, 5, precision, compare_models(modl2, modl5, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(2, 6, precision, compare_models(modl2, modl6, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(2, 7, precision, compare_models(modl2, modl7, precision)))

log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(3, 4, precision, compare_models(modl3, modl4, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(3, 5, precision, compare_models(modl3, modl5, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(3, 6, precision, compare_models(modl3, modl6, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(3, 7, precision, compare_models(modl3, modl7, precision)))

log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(4, 5, precision, compare_models(modl4, modl5, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(4, 6, precision, compare_models(modl4, modl6, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(4, 7, precision, compare_models(modl4, modl7, precision)))

log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(5, 6, precision, compare_models(modl5, modl6, precision)))
log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(5, 7, precision, compare_models(modl5, modl7, precision)))

log.write("Compare the state of model {}/{} (precision = {}): {}\n".format(6, 7, precision, compare_models(modl6, modl7, precision)))
log.close()
