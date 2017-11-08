"""
Aggregate experience tuple into pytorch-ready tensors
"""
import numpy as np
from easydict import EasyDict
import torch
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable


def _obs_concat(obs_list):
    # convert uint8 to float32, if any
    return Variable(U.to_float_tensor(np.stack(obs_list)))


def torch_aggregate(exp_list, obs_spec, action_spec):
    # TODO add support for more diverse obs_spec and action_spec
    """

    Args:
        exp_list:
        obs_spec:
        action_spec:

    Returns:

    """
    U.assert_type(obs_spec, dict)
    U.assert_type(action_spec, dict)
    obses0, actions, rewards, obses1, dones = [], [], [], [], []
    for exp in exp_list:
        obses0.append(np.array(exp['obses'][0], copy=False))
        actions.append(exp['action'])
        rewards.append(exp['reward'])
        obses1.append(np.array(exp['obses'][1], copy=False))
        dones.append(float(exp['done']))
    if action_spec['type'] == 'continuous':
        actions = _obs_concat(actions)
    elif action_spec['type'] == 'discrete':
        actions = Variable(torch.LongTensor(actions).unsqueeze(1))
    else:
        raise NotImplementedError('action_spec unsupported '+str(action_spec))
    return EasyDict(
        obses=[_obs_concat(obses0), _obs_concat(obses1)],
        actions=actions,
        rewards=Variable(torch.FloatTensor(rewards).unsqueeze(1)),
        dones=Variable(torch.FloatTensor(dones).unsqueeze(1)),
    )