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
from tensorboardX import SummaryWriter

#from ipdb import set_trace as pdb


random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load Hyperparameters
parser = argparse.ArgumentParser()
#parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
parser.add_argument('--use-mc', action='store_true', help='whether to use memcache')
parser.add_argument('--resume', type=str, default=None, help='whether to resume, [best, latest, epoch]')
parser.add_argument('--test-only', action='store_true', help='whether to test only')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--base_lr', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
args = parser.parse_args()

if args.use_mc:
    import mc
    import io

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(args.decay_step)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_MIN = 0.00001
        
prefix = '../../../../pcnn_mj_dataset/pcd_npy/'
datalist_path = prefix + 'train_datalist1000.txt'
labellist_path = prefix + 'train_labellist1000.txt'
test_datalist_path = prefix + 'test_datalist200.txt'
test_labellist_path = prefix + 'test_labellist200.txt'
num_class = int(open(labellist_path, 'r').readline().strip())

save_dir = '../../../../exp/pcnn/'
os.makedirs(join(save_dir, 'checkpoint'), exist_ok=True)
os.makedirs(join(save_dir, 'log'), exist_ok=True)


def save_model(model, epoch, global_step, accuracy):
    ckpt = {
        'epoch': epoch,
        'global_step': global_step,
        'accuracy': accuracy,
        'state_dict': model.state_dict(),
    }
    path = join(save_dir, 'checkpoint', '{}.ckpt.pth'.format(epoch))
    torch.save(ckpt, path)
    path = join(save_dir, 'checkpoint', 'latest.ckpt.pth'.format(epoch))
    torch.save(ckpt, path)
    if exists(join(save_dir, 'checkpoint', 'best.ckpt.pth')):
        path = join(save_dir, 'checkpoint', 'best.ckpt.pth'.format(epoch))
        best_ckpt = torch.load(path)
        best_acc = best_ckpt['accuracy']
        if accuracy > best_acc:
            torch.save(ckpt, path)
    else:
        path = join(save_dir, 'checkpoint', 'best.ckpt.pth'.format(epoch))
        torch.save(ckpt, path)
    print('Epoch {}: model saved'.format(epoch))


def load_model(model, ckpt_name):
    path = join(save_dir, 'checkpoint', ckpt_name)
    def map_func(storage, location):
        return storage.cuda()
    assert exists(path)
    ckpt = torch.load(path, map_location=map_func)

    global global_step, start_epoch
    global_step = ckpt['global_step']
    start_epoch = ckpt['epoch'] + 1
    model.load_state_dict(ckpt['state_dict'])
    print('Ckpt loaded: {}, global step : {}, current epoch {}'.format(ckpt_name, global_step, start_epoch))


def test_model(model):
    print('Start testing')
    model.eval()
    test_dataset = CustomDataset(datalist_path=test_datalist_path, labellist_path=test_labellist_path, prefix=prefix)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 drop_last=False)
    total_acc = 0, 0  # count, total number
    for batch_idx, (data, label) in enumerate(test_dataloader):
        if args.gpu:
            data, label = data.cuda(), label.cuda()
        P_sampled = data
        out = model((P_sampled, P_sampled))

        _, pred = out.max(dim=1)
        acc_cnt = (pred == label).sum().item()
        total_acc = total_acc[0] + acc_cnt, total_acc[1] + data.shape[0]
        if batch_idx % 25 == 0:
            print('Testing: Epoch {} iter {}: acc {:.4f}'.format(epoch, batch_idx, acc_cnt / data.shape[0]))
            tb_logger.add_scalar('test_batch_acc', acc_cnt / data.shape[0])
    test_acc = total_acc[0] / total_acc[1]
    print('Testing: Epoch {} done: acc {:.4f}'.format(epoch, test_acc))


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
        self.initialized = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path = os.path.join(self.prefix, self.data[index])

        if args.use_mc:
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(data_path, value)
            value_str = mc.ConvertBuffer(value)
            buff = io.BytesIO(value_str)
            data = np.load(buff)
        else:
            data = np.load(data_path)

        data = data.astype('float32')
        if self.transform is not None:
            data = self.transform(data)
        return data, self.labels[index]

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)
            self.initialized = True


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


tb_logger = SummaryWriter(join(save_dir, 'events'))
ftrain_epoch_acc = open(join(save_dir, 'log', 'train_epoch_acc.txt'), 'a+')
ftrain_batch_acc = open(join(save_dir, 'log', 'train_batch_acc.txt'), 'a+')
ftest_epoch_acc = open(join(save_dir, 'log', 'test_epoch_acc.txt'), 'a+')

print("------Building model-------")
model = Classifier()
if args.gpu:
    model = model.cuda()
print("------Successfully Built model-------")

global global_step, start_epoch
global_step = 1
start_epoch = 1

if args.resume is not None:
    load_model(model, args.resume + '.ckpt.pth')

if args.test_only:
    test_model(model)
    exit(0)

lr = args.base_lr * args.decay_rate ** (global_step // args.decay_step)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
loss_fn = nn.CrossEntropyLoss()
print('====Start training: epoch {}, global_step {}, lr {}'.format(start_epoch, global_step, lr))
for epoch in range(start_epoch, args.max_epoch+1):
    if epoch > 1 and global_step % args.decay_step == 0:
        lr = args.base_lr * args.decay_rate ** (global_step // args.decay_step)
        lr = max(lr, LEARNING_RATE_MIN)
        print("NEW LEARNING RATE:", lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)

    model.train()
    train_dataset = CustomDataset(datalist_path=datalist_path, labellist_path=labellist_path, prefix=prefix)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                 drop_last=True)
    total_acc = 0, 0  # count, total number
    for batch_idx, (data, label) in enumerate(train_dataloader):
        if args.gpu:
            data, label = data.cuda(), label.cuda()
        P_sampled = data
        out = model((P_sampled, P_sampled))
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = out.max(dim=1)
        acc_cnt = (pred == label).sum().item()
        total_acc = total_acc[0] + acc_cnt, total_acc[1] + data.shape[0]
        if global_step % 25 == 0:
            print('Epoch {} iter {}: loss {}, acc {:.4f}'.format(epoch, batch_idx, loss.item(), acc_cnt/data.shape[0]))
            ftrain_batch_acc.write('Epoch {} iter {}: {}\n'.format(epoch, batch_idx, acc_cnt/data.shape[0]))
            tb_logger.add_scalar('train_batch_acc', acc_cnt/data.shape[0], global_step)
        global_step += 1
    train_acc = total_acc[0]/total_acc[1]
    print('Epoch {} done: acc {:.4f}'.format(epoch, train_acc))
    ftrain_epoch_acc.write('Epoch {}: {}\n'.format(epoch, train_acc))
    tb_logger.add_scalar('train_epoch_acc', train_acc, epoch)

    if epoch % 1 == 0:
        print('Start testing')
        model.eval()
        test_dataset = CustomDataset(datalist_path=test_datalist_path, labellist_path=test_labellist_path, prefix=prefix)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                      drop_last=False)
        total_acc = 0, 0  # count, total number
        for batch_idx, (data, label) in enumerate(test_dataloader):
            if args.gpu:
                data, label = data.cuda(), label.cuda()
            P_sampled = data
            out = model((P_sampled, P_sampled))

            _, pred = out.max(dim=1)
            acc_cnt = (pred == label).sum().item()
            total_acc = total_acc[0] + acc_cnt, total_acc[1] + data.shape[0]
            if batch_idx % 25 == 0:
                print('Testing: Epoch {} iter {}: acc {:.4f}'.format(epoch, batch_idx, acc_cnt / data.shape[0]))
                tb_logger.add_scalar('test_batch_acc', acc_cnt/data.shape[0])
        test_acc = total_acc[0]/total_acc[1]
        print('Testing: Epoch {} done: acc {:.4f}'.format(epoch, test_acc))
        ftest_epoch_acc.write('Epoch {}: {}\n'.format(epoch, test_acc))
        tb_logger.add_scalar('test_epoch_acc', test_acc)

        save_model(model, epoch, global_step, test_acc)
    ftrain_epoch_acc.flush(), ftrain_batch_acc.flush(), ftest_epoch_acc.flush()

"""
TODO:
transform: sample, shuffle, jitter,
"""

#
