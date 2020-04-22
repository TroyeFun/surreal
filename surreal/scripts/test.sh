#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 
folder=../../exp/subproc/sawyerpickplace_run2/
python main/rollout.py --folder $folder --checkpoint learner.37000.ckpt --algo ppo  \
    --record --record-folder $folder/rollout/ --record-every 10
