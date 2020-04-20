#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python subproc/surreal_subproc.py sawyerpickplace_obj-state -al ppo -na 4 --env robosuite:SawyerPickPlace
