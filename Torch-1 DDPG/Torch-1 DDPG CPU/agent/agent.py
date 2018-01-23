"""@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: agent.py
@time: 2018/1/17 9:25
"""
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as nf
from torch.autograd import Variable
from torch.optim import Adam

from agent.memory import Memory
from agent.forward import ActorNet, CriticNet
from agent.utils import Noise
from config import *


def soft_update(target, source, tau=1e-3):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)


class Agent(object):
    def __init__(self, state_size, action_size, action_limits=1.):
        self.state_size = state_size
        self.action_size = action_size
        self.action_limits = action_limits

        self.memory = Memory(MEMORY_SIZE)
        self.noise = Noise(action_size)

        self.actor = ActorNet(state_size, action_size)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = Adam(
            self.actor.parameters(), LEARNING_RATE)

        self.critic = CriticNet(state_size, action_size)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = Adam(
            self.critic.parameters(), LEARNING_RATE)

    def append(self, *args):
        self.memory.append(*args)

    def sample(self, *args):
        return self.memory.sample(*args)

    def get_exploitation_policy(self, state):
        state = Variable(torch.from_numpy(np.float32(state)))
        action = self.target_actor(state).detach()
        return action.data.numpy()

    def get_exploration_policy(self, state):
        state = Variable(torch.from_numpy(np.float32(state)))
        action = self.actor(state).detach()
        return action.data.numpy() + \
               (self.noise() * self.action_limits)

    def optimize(self, batch_size=BATCH_SIZE):
        batch = self.sample(batch_size)
        state, action, reward,  next_state =\
            [Variable(torch.from_numpy(i)) for i in batch]

        next_action = self.target_actor.forward(next_state).detach()
        next_value = torch.squeeze(
            self.target_critic(next_state, next_action).detach())
        target_value = reward + GAMMA * next_value
        value = torch.squeeze(self.critic(state, action))

        loss_critic = nf.smooth_l1_loss( value, target_value)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        policy_action = self.actor(state)
        loss_actor = -1 * torch.sum(self.critic(state, policy_action))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

    def restore_models(self, num_episode):
        self.actor.load_state_dict(torch.load(
            "./Models/{}_actor.pkl".format(num_episode)))
        self.critic.load_state_dict(torch.load(
            "./Models/{}_critic.pkl".format(num_episode)))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def save_models(self, num_episode):
        torch.save(self.target_actor.state_dict(),
                   "actor_{}.pkl".format(num_episode))
        torch.save(self.target_critic.state_dict(),
                   "critic_{}.pkl".format(num_episode))
        print('Models saved successfully')