#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 
#surreal-subproc test -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_obj-state -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_run2 -al ppo -na 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/subproc/sawyerpickplace_run2/checkpoint/


#surreal-tmux create spp_ppo_na4_pix --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/tmux/spp-ppo-na4-pix/checkpoint/
#surreal-tmux create spp_ppo_na4_nopix_run2 --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace  -- --restore-folder ../../exp/tmux/spp-ppo-na4-nopix-run2/checkpoint/
#surreal-tmux create sppsgl_ppo_na4_nopix --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlaceSingle  -- --restore-folder ../../exp/tmux/sppsgl-ppo-na4-nopix/checkpoint/

#surreal-tmux create sppsgl_ppo_na6_nopix --algorithm ppo --num_agents 6 --env robosuite:SawyerPickPlaceSingle -- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix/checkpoint/
#surreal-tmux create sppsgl_ddpg_na6_nopix --algorithm ddpg --num_agents 6 --env robosuite:SawyerPickPlaceSingle  -- --restore-folder ../../exp/tmux/sppsgl-ddpg-na6-nopix/checkpoint/
#surreal-tmux create sppsgl_ppo_na6_nopix_10hz --algorithm ppo --num_agents 6 --num_evals 1 --env robosuite:SawyerPickPlaceSingle -- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix-10hz/checkpoint/
surreal-tmux create sbl_ddpg_na2_nopix_1hz --algorithm ddpg --num_agents 2 --num_evals 0 --env robosuite:SawyerLift #-- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix-10hz/checkpoint/

