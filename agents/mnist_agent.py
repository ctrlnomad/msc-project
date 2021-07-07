from typing import Tuple
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from causal_env.envs import Timestep
class BayesianConvNet(nn.Module):
    """
    should have two brancches for two treatment effects
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.view(-1, 320)
        x = torch.relu(self.fc1(x))
        # two separate confounders? 
        x = torch.dropout(x, training=True) # TODO: does just this make it a bayesian network?
        x = self.fc2(x)

    def compute_uncertainty(self, x, n_samples):
        # compute entropy with dropout
        bs = len(x)
        result = torch.zeros((bs, n_samples))
        for i in range(n_samples):
            result[i] = self(x)
        return result.var(dim=-1)


class AgentConfig:
    dim_in: Tuple[int] = (28, 28)
    memsize: int = 100_000
    mc_dropout_samples: int = 100

class CmnistBanditAgent:
    """
    ITE network p(r | x, t, w). The observational data is confounded by other treatments
    ITE or CATE? should I ask?
    and variance that is not fixed 
    
    loss = -dist.normal(mu_theta(x), sigma_theta(x)).log_prob(y)

    half the round we learn by empty interventions
    the other times we learn by choosing based on epsitemic uncertainty
    we evaluate by regret. 
    """
    def __init__(self, config: AgentConfig):
        # learning is the ITE of causal arms
        self.memory = deque(maxlen=config.memsize)
        self.possible_models = []

        # needs to determine ITE
        
        # p(r | x, t, w) for each of the digits
        self.effect_estimator = BayesianConvNet()
        self.n_samples = config.mc_dropout_samples

    def train(self, n_epoch):
        pass

    def calculate_uncertainties(self, contexts: torch.Tensor):
        variances = self.effect_estimator.compute_uncertainty(contexts, self.n_samples)
        return variances

    def remember(self, timestep: Timestep):
        self.memory.append(timestep)


    
    