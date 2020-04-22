#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 
#surreal-subproc test -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_obj-state -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_run2 -al ppo -na 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/subproc/sawyerpickplace_run2/checkpoint/


surreal-tmux create spp_ppo_na4_pix --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/subproc/sawyerpickplace_run2/checkpoint/
