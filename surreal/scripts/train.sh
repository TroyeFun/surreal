#!/bin/bash
#CUDA_VISIBLE_DEVICES=2 python subproc/surreal_subproc.py test -al ppo -na 4 --env robosuite:SawyerPickPlace
#CUDA_VISIBLE_DEVICES=0 python subproc/surreal_subproc.py sawyerpickplace_obj-state -al ppo -na 4 --env robosuite:SawyerPickPlace
CUDA_VISIBLE_DEVICES=2 surreal-subproc sawyerpickplace_run2 -al ppo -na 4 --env robosuite:SawyerPickPlace -- --restore-folder ../../exp/subproc/sawyerpickplace_run2/checkpoint/
