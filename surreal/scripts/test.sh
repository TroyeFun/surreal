#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python main/rollout.py --folder ../../exp/subproc/sawyerpickplace_run2/ --checkpoint learner.37000.ckpt --algo ppo  \
    --record --record-folder ../../exp/subproc/sawyerpickplace_run2/rollout/ --record-every 10
