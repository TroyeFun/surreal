#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0 
#surreal-subproc test -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_obj-state -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_run2 -al ppo -na 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/subproc/sawyerpickplace_run2/checkpoint/


#surreal-tmux create spp_ppo_na4_pix --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/tmux/spp-ppo-na4-pix/checkpoint/
#surreal-tmux create spp_ppo_na4_nopix_run2 --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace  -- --restore-folder ../../exp/tmux/spp-ppo-na4-nopix-run2/checkpoint/
#surreal-tmux create test --algorithm ppo --num_agents 1 --env robosuite:SawyerPickPlaceTarget
surreal-subproc test --algorithm ddpg --num_agents 1 --num_evals 1 --env robosuite:SawyerPickPlaceSingleMultiTaskTarget -- --restore-folder /Users/fanghongyu/Desktop/Black_Hole/material/I_love_study/实验室/毕设/experiment/exp/tmux/sppsglmttct-pick-ddpg-na8-nopix/checkpoint/
