#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 
folder=../../exp/tmux/sbl-ddpg-na2-nopix-1hz/
python main/rollout.py --folder $folder --checkpoint learner.1800000.ckpt --algo ddpg  \
    --record --record-folder $folder/rollout/ --record-every 1 --episode-limit 3
