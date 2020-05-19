import pickle
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, required=True)
args = parser.parse_args()

with open(args.path, 'rb') as f:
    ckpt = pickle.load(f)

torch.save(ckpt, args.path + '.cpu')
ckpt = torch.load(args.path + '.cpu', map_location='cpu')
torch.save(ckpt, args.path + '.cpu')

