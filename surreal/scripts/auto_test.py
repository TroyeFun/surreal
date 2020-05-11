#!/bin/bash
import sys
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, required=True)
parser.add_argument('-s', '--start', type=int, default=0)
parser.add_argument('--step', type=int, default=100000)  # for ddpg
parser.add_argument('--record-every', type=int, default=1)  
parser.add_argument('-n', '--num-record', type=int, default=2)  
args = parser.parse_args()

if args.start == 0:
    args.start = args.step
ckpt = args.start
while True:
    if not os.path.exists(os.path.join(args.folder, 'checkpoint', 'learner.{}.ckpt'.format(ckpt))):
        time.sleep(600) # 10min
    else:
        cmd = f"python main/rollout.py --folder {args.folder} --checkpoint learner.{ckpt}.ckpt --algo ddpg  --record --record-folder {os.path.join(args.folder, 'rollout', str(ckpt))} --record-every {args.record_every} --episode-limit {args.record_every*args.num_record}"
        print(cmd)
        os.system(cmd)
        print(ckpt, 'ckpt tested.')
        ckpt += args.step


