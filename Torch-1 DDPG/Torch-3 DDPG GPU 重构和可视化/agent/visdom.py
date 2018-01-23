"""
@author: Young
@license: (C) Copyright 2013-2017
@contact: aidabloc@163.com
@file: visdom.py
@time: 2018/1/18 22:03
"""
import numpy as np
from collections import namedtuple
import visdom

vis = visdom.Visdom()
EpisodeStats = namedtuple("Stats",
                          ["episode_lengths", "episode_rewards", "mean_rewards"])


def episode_stats(stats):
    vis.line(X=np.arange(len(stats.mean_rewards)),
             Y=np.array(stats.mean_rewards),
             win="DDPG Mean Reward (100 episodes)",
             opts=dict(title=("DDPG Mean Reward (100 episodes)"),
                       ylabel="MEAN REWARD (100 episodes)",
                       xlabel="Episode"))

    vis.line(X=np.cumsum(stats.episode_lengths),
             Y=np.arange(len(stats.episode_lengths)),
             win="DDPG Episode Per Time Step",
             opts=dict( title=("DDPG Episode per time step"),
                        ylabel="Episode",
                        xlabel="Time Steps"))
