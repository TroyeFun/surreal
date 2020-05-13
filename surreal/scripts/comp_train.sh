#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
#surreal-subproc test -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_obj-state -al ppo -na 4 --env robosuite:SawyerPickPlace
#surreal-subproc sawyerpickplace_run2 -al ppo -na 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/subproc/sawyerpickplace_run2/checkpoint/


#surreal-tmux create spp_ppo_na4_pix --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/tmux/spp-ppo-na4-pix/checkpoint/
#surreal-tmux create spp_ppo_na4_nopix_run2 --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlace  -- --restore-folder ../../exp/tmux/spp-ppo-na4-nopix-run2/checkpoint/
#surreal-tmux create sppsgl_ppo_na4_nopix --algorithm ppo --num_agents 4 --env robosuite:SawyerPickPlaceSingle  -- --restore-folder ../../exp/tmux/sppsgl-ppo-na4-nopix/checkpoint/

#surreal-tmux create sppsgl_ppo_na6_nopix --algorithm ppo --num_agents 6 --env robosuite:SawyerPickPlaceSingle -- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix/checkpoint/
#surreal-tmux create sppsgl_ddpg_na6_nopix --algorithm ddpg --num_agents 6 --env robosuite:SawyerPickPlaceSingle  -- --restore-folder ../../exp/tmux/sppsgl-ddpg-na6-nopix/checkpoint/
#MUJOCO_PY_SKIP_ACTIVATE=1 \

LOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 MC_COUNT_DISP=1000000 \
OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0  \
srun --mpi=pmi2 --job-name tt --partition=VA -n1 --gres=gpu:2 --ntasks-per-node=2  -w SH-IDC1-10-5-37-51 \
python -u subproc/surreal_subproc.py sbl-ddpg-na4-pix-fs1-run2 --algorithm ddpg --num_agents 4 --num_evals 1 --env robosuite:SawyerLift #-- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix-10hz/checkpoint/
#python -u subproc/surreal_subproc.py sppsglmtt-ddpg-na4-nopix-fs1 --algorithm ddpg --num_agents 4 --num_evals 1 --env robosuite:SawyerPickPlaceMultiTaskTarget #-- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix-10hz/checkpoint/
#python -u subproc/surreal_subproc.py sbl-ddpg-na4-pix-fs1 --algorithm ddpg --num_agents 4 --num_evals 1 --env robosuite:SawyerLift #-- --restore-folder ../../exp/tmux/sppsgl-ppo-na6-nopix-10hz/checkpoint/

