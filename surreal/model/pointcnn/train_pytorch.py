import argparse 
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists, join
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import math
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


from utils.model import RandPointCNN
from utils.util_funcs import knn_indices_func
from utils.util_layers import Dense

from ipdb import set_trace as pdb


random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load Hyperparameters
parser = argparse.ArgumentParser()
#parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=2, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
args = parser.parse_args()

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(args.decay_step)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_MIN = 0.00001
        
prefix = '../../../../pcnn_mj_dataset/pcd_npy/'
datalist_path = prefix + 'train_datalist.txt'
labellist_path = prefix + 'train_labellist.txt'
num_class = int(open(labellist_path, 'r').readline().strip())

save_dir = '../../../../exp/pcnn/'
os.makedirs(join(save_dir, 'checkpoint'), exist_ok=True)

class CustomDataset(Dataset):

    def __init__(self, datalist_path, labellist_path, prefix, transform=None):
        self.data = []
        self.labels = []
        self.prefix = prefix
        self.transform = transform
        fdata = open(datalist_path, 'r')
        flabel = open(labellist_path, 'r')
        for data, label in zip(fdata.readlines(), flabel.readlines()[1:]):
            self.data.append(data.strip())
            self.labels.append(int(label.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path = os.path.join(self.prefix, self.data[index])
        data = np.load(data_path).astype('float32')
        if self.transform is not None:
            data = self.transform(data)
        return data, self.labels[index]

# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, num_class, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        x = self.pcnn2(x)[1]  # grab features
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


print("------Building model-------")
model = Classifier()
if args.gpu:
    model = model.cuda()
print("------Successfully Built model-------")


lr = args.learning_rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
loss_fn = nn.CrossEntropyLoss()

global_step = 1
for epoch in range(1, args.max_epoch+1):
    if epoch > 1:
        lr *= args.decay_rate ** (global_step // args.decay_step)
        if lr > LEARNING_RATE_MIN:
            print("NEW LEARNING RATE:", lr)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

    train_dataset = CustomDataset(datalist_path=datalist_path, labellist_path=labellist_path, prefix=prefix)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                 drop_last=True)

    for batch_idx, (data, label) in enumerate(train_dataloader):
        P_sampled = data
        out = model((P_sampled, P_sampled))
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 25 == 0:
            print('Epoch {} iter {}: loss {}'.format(epoch, batch_idx, loss.item()))
        global_step += 1

"""
TODO:
accuracy
test
save, load
tensorboard
transform: sample, shuffle, jitter,
"""

#
