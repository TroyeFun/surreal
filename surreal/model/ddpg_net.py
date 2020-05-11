from queue import Queue
import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np
import itertools
import torchx.nn as nnx

from .model_builders import *
from .z_filter import ZFilter
from .pointcnn.utils.model import PCNNStemNetwork
from .pointcnn.utils.data_utils import Pix2PCD
from ipdb import set_trace as pdb

class DDPGModel(nnx.Module):

    def __init__(self,
                 obs_spec,
                 action_dim,
                 model_config,
                 use_cuda,
                 critic_only=False,
                 if_pixel_input=False,
                 if_pcd_input=False,
                 ):
        super(DDPGModel, self).__init__()

        self.obs_spec = obs_spec
        self.action_dim = action_dim
        self.model_config = model_config
        self.use_cuda = use_cuda
        self.critic_only = critic_only

        use_layernorm = model_config.use_layernorm
        actor_fc_hidden_sizes = model_config.actor_fc_hidden_sizes
        critic_fc_hidden_sizes = model_config.critic_fc_hidden_sizes
        conv_out_channels = model_config.conv_spec.out_channels
        conv_kernel_sizes = model_config.conv_spec.kernel_sizes
        conv_strides = model_config.conv_spec.strides
        conv_hidden_dim = model_config.conv_spec.hidden_output_dim

        # hyperparameters
        self.if_pixel_input = if_pixel_input
        self.if_pcd_input = if_pcd_input
        self.action_dim = action_dim
        self.use_layernorm = use_layernorm

        #if self.if_pixel_input:
        #    self.input_dim = obs_spec['pixel']['camera0']
        #else:
        #    self.input_dim = obs_spec['low_dim']['flat_inputs'][0]

        concatenated_perception_dim = 0
        if self.if_pixel_input:
            self.cnn_stem = CNNStemNetwork(obs_spec['pixel']['camera0'], conv_hidden_dim, conv_channels=conv_out_channels,
                                             kernel_sizes=conv_kernel_sizes, strides=conv_strides)
            concatenated_perception_dim += conv_hidden_dim
        
        if self.if_pcd_input:
            self.pix2pcd = Pix2PCD(self.obs_spec['env_info']['camera_mat'], 
                                  self.obs_spec['env_info']['camera_pos'],
                                  self.obs_spec['env_info']['camera_f'],
                                  self.obs_spec['pixel']['camera0'],
                                  use_cuda)
            self.pcnn_stem = PCNNStemNetwork(self.model_config.pcnn_feature_dim)
            if use_cuda:
                self.pcnn = self.pcnn.cuda()
            concatenated_perception_dim += self.model_config.pcnn_feature_dim
        

        if 'low_dim' in obs_spec:
            concatenated_perception_dim += obs_spec['low_dim']['flat_inputs'][0]
        if not critic_only:
            self.actor = ActorNetworkX(concatenated_perception_dim, self.action_dim, hidden_sizes=actor_fc_hidden_sizes,
                                       use_layernorm=self.use_layernorm)
        else:
            self.actor = None
        self.critic = CriticNetworkX(concatenated_perception_dim, self.action_dim, hidden_sizes=critic_fc_hidden_sizes,
                                     use_layernorm=self.use_layernorm)

    def get_actor_parameters(self):
        return itertools.chain(self.actor.parameters())

    def get_critic_parameters(self):
        params = self.critic.parameters()
        if self.if_pixel_input:
            params = itertools.chain(params, self.cnn_stem.parameters())
        if self.if_pcd_input:
            params = itertools.chain(params, self.pcnn_stem.parameters())
        return params

    def clear_actor_grad(self):
        self.actor.zero_grad()

    def clear_critic_grad(self):
        self.critic.zero_grad()
        if self.if_pixel_input:
            self.cnn_stem.zero_grad()
        if self.if_pcd_input:
            self.pcnn_stem.zero_grad()

    def update_target_params(self, net, update_type, tau=1):
        if update_type == 'soft':
            if not self.critic_only:
                self.actor.soft_update(net.actor, tau)
            self.critic.soft_update(net.critic, tau)
            if self.if_pixel_input:
                self.cnn_stem.soft_update(net.cnn_stem, tau)
            if self.if_pcd_input:
                self.pcnn_stem.soft_update(net.pcnn_stem, tau)
        elif update_type == 'hard':
            if not self.critic_only:
                self.actor.load_state_dict(net.actor.state_dict())
            self.critic.load_state_dict(net.critic.state_dict())
            if self.if_pixel_input:
                self.cnn_stem.load_state_dict(net.cnn_stem.state_dict())
            if self.if_pcd_input:
                self.pcnn_stem.load_state_dict(net.pcnn_stem.state_dict())

    def forward_actor(self, obs):
        return self.actor(obs)

    def forward_critic(self, obs, action):
        return self.critic(obs, action)

    def forward_perception(self, obs):
        concatenated_inputs = []
        if self.if_pixel_input:
            obs_pixel = obs['pixel']['camera0']
            obs_pixel = self.scale_image(obs_pixel)
            cnn_updated = self.cnn_stem(obs_pixel)
            concatenated_inputs.append(cnn_updated)

        if self.if_pcd_input:
            obs_pcd = self.pix2pcd(obs['pixel']['camera0'], obs['env_info']['target_color']) 
            obs_pcd = self.pcnn_stem(obs_pcd)
            concatenated_inputs.append(obs_pcd)

        if 'low_dim' in obs:
            concatenated_inputs.append(obs['low_dim']['flat_inputs'])
        concatenated_inputs = torch.cat(concatenated_inputs, dim=1)
        return concatenated_inputs

    def forward(self, obs_in, calculate_value=True, action=None):
        obs_in = self.forward_perception(obs_in)
        if action is None:
            action = self.forward_actor(obs_in)
        value = None
        if calculate_value:
            value = self.forward_critic(obs_in, action)
        return action, value

    def scale_image(self, obs, scaling_factor=255.0):
        '''
        Given uint8 input from the environment, scale to float32 and
        divide by 255 to scale inputs between 0.0 and 1.0
        '''
        return obs / scaling_factor
        
