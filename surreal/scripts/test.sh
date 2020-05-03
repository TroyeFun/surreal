#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 
folder=../../exp/tmux/sbl-ddpg-na2-nopix-1hz/
python main/rollout.py --folder $folder --checkpoint learner.400000.ckpt --algo ddpg  \
    --record --record-folder $folder/rollout/ --record-every 2 --episode-limit 4
