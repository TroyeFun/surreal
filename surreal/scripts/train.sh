#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 
#surreal-tmux create sblbb_ddpg_na4_nopix_1hz --algorithm ddpg --num_agents 2 --num_evals 0 --env robosuite:SawyerLiftBinobs #-- --restore-folder ../../exp/tmux/sppsglmtt-ddpg-na4-nopix-1hz/checkpoint/
#surreal-tmux create sblbo_ddpg_na3_nopix_nowall --algorithm ddpg --num_agents 3 --num_evals 1 --env robosuite:SawyerLiftBinobs #-- --restore-folder ../../exp/tmux/sppsglmtt-ddpg-na4-nopix-1hz/checkpoint/
#surreal-tmux create sbl_ddpg_na3_nopix --algorithm ddpg --num_agents 2 --num_evals 1 --env robosuite:SawyerLiftRandom -- --restore-folder ../../exp/tmux/sbl-ddpg-na3-nopix/checkpoint/
#surreal-tmux create sbl_ddpg_na4_pcd_fs1 --algorithm ddpg --num_agents 4 --num_evals 1 --env robosuite:SawyerLift #-- --restore-folder ../../exp/tmux/sppsglmtt-ddpg-na4-nopix-1hz/checkpoint/
surreal-tmux create sppsglmtt_ddpg_na2_nopix --algorithm ddpg --num_agents 2 --num_evals 1 --env robosuite:SawyerPickPlaceSingleMultiTaskTarget -- --restore-folder ../../exp/tmux/sppsglmtt-ddpg-na2-nopix/checkpoint/
