"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc # 标准库，用于定义抽象基类
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers) # 把层序列化并返回一个可训练的MLP模块
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta): # BasePolicy是一个抽象基类，定义了策略的接口
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp( # 动作均值网络
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter( # 动作标准差网络

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()), # 串联mean_net和logstd参数
            self.learning_rate
        )

    def save(self, filepath): # 保存模型参数
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        # raise NotImplementedError

        # 确保 observation 在正确 device 并为 tensor
        if not isinstance(observation, torch.Tensor):
            obs = ptu.from_numpy(observation)
        else:
            obs = observation.to(ptu.device)

        mean = self.mean_net(obs)  # (batch, ac_dim) 或 (ac_dim,)
        # logstd shape: (ac_dim,), 需要 broadcast 到 mean 的形状
        std = torch.exp(self.logstd)
        if mean.dim() == 2:
            std = std.unsqueeze(0).expand_as(mean)
        # 构造可重参数化的正态分布并采样
        dist = distributions.Normal(mean, std)
        action = dist.rsample()  # 可求导的采样
        return action, dist

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        # loss = TODO

        # 准备张量并移动到 device
        if not isinstance(observations, torch.Tensor):
            obs_tensor = ptu.from_numpy(observations)
        else:
            obs_tensor = observations.to(ptu.device)

        if not isinstance(actions, torch.Tensor):
            actions_tensor = ptu.from_numpy(actions)
        else:
            actions_tensor = actions.to(ptu.device)

        # 前向得到均值（不需要采样用于 MSE）
        pred_mean = self.mean_net(obs_tensor)

        # 均方误差损失
        loss = F.mse_loss(pred_mean, actions_tensor)

        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return{
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
