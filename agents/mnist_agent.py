from typing import Tuple
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from causal_env.envs import Timestep

def mlp(inp):
    return nn.Sequential([
            nn.Linear(inp, 50),
            nn.LeakyReLU(),
            nn.Dropout(p=1/4),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Dropout(p=1/4),
            nn.Linear(25, 2), # mu and sigma

        ])

class BayesianConvNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_block = nn.Sequential([
            nn.Conv2d(1, 10, kernel_size=5),
            nn.Dropout2d(),
            nn.Conv2d(10, 20, kernel_size=5)
        ])

        self.treatment_effect = mlp(320)
        self.no_treatment_effect = mlp(320)

    def forward(self, x, t: int):
        x = self.conv_block(x)
        x = torch.view(-1, 320)

        if t:
            r_dist = self.treatment_effect(x)
        else:
            r_dist = self.no_treatment_effect(x)

        return r_dist
        

    def compute_uncertainty(self, x,  n_samples):
        # compute entropy with dropout
        bs = len(x)
        result = torch.zeros((bs, 2, n_samples)) # two for binary treatment
        for i in range(n_samples):
            result[..., 0 , i] = self(x, 0)[0]
            result[..., 1 , i] = self(x, 1)[0] 
        return result.var(dim=-1) # Var[E[Y | X]]


class AgentConfig:
    dim_in: Tuple[int] = (28, 28)
    memsize: int = 100_000
    mc_dropout_samples: int = 100

    cuda: bool = True

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
        self.gpu = config.cuda
        if self.gpu:
            self.effect_estimator.cuda()
        self.n_samples = config.mc_dropout_samples

    def train(self, n_epoch):

        batch_timestep = Timestep(*zip(self.memory))
        # to dataset, dataloader
        # to batches 
        # nll loss and adam

    def calculate_uncertainties(self, contexts: torch.Tensor):
        variances = self.effect_estimator.compute_uncertainty(contexts, self.n_samples)
        return variances

    def remember(self, timestep: Timestep):
        self.memory.append(timestep)


    
    