from typing import Tuple
from dataclasses import dataclass
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.distributions as distributions

from causal_env.envs import Timestep, TimestepDataset

def mlp(inp):
    return nn.Sequential([
            nn.Linear(inp, 50),
            nn.LeakyReLU(),
            nn.Dropout(p=1/4),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Dropout(p=1/4),
            nn.Linear(25, 4), # mu and sigma for treatment and non treatment

        ])

class BayesianConvNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_block = nn.Sequential([
            nn.Conv2d(1, 10, kernel_size=5),
            nn.Dropout2d(),
            nn.Conv2d(10, 20, kernel_size=5)
        ])

        self.head = mlp(320)

    def forward(self, x, t):
        x = self.conv_block(x)
        x = torch.view(-1, 320)

        r_dist = self.head(x)
        r_dist = r_dist.view(-1, 2, 2)

        return r_dist
        

    def compute_uncertainty(self, x,  n_samples):
        # compute entropy with dropout
        bs = len(x)
        result = torch.zeros((bs, 2, n_samples)) # two for binary treatment
        for i in range(n_samples):
            result[..., 0 , i] = self(x, 0)[0]
            result[..., 1 , i] = self(x, 1)[0] 
        return result.var(dim=-1) # Var[E[Y | X]]

@dataclass
class AgentConfig:
    dim_in: Tuple[int] = (28, 28)
    memsize: int = 100_000
    mc_dropout_samples: int = 100

    do_nothing: float = 0.5 # do nothing for this many steps
    cuda: bool = True


@dataclass
class History:
    loss: List[float] = []


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
        self.config = config

        self.history = History()
        
        if self.config.gpu:
            self.effect_estimator.cuda()


    def train_once(self, ):
        dataset = TimestepDataset.build(self.memory)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        criterion = nn.NLLLoss()

        opt = optim.Adam(self.effect_estimator.parameters())

        for contexts, effects, treatments in loader:
            opt.zero_grad()
            
            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments
            
            pred_effects = self.effect_estimator(contexts, treatments)

            pred_effects = pred_effects[:, treatments, :]
            mu_pred = pred_effects[:, 0]
            sigma_pred = pred_effects[:, 1]

            bs = pred_effects.size(0)

            sigma_mat_pred = torch.zeros(bs)
            sigma_mat_pred[torch.eye(bs)] = sigma_pred

            loss = -distributions.MultivariateNormal(mu_pred, sigma_mat_pred).log_prob(effects).mean()
            loss.backward()
            opt.step()

            self.history.loss.append(loss.item())

    def calculate_uncertainties(self, contexts: torch.Tensor):
        variances = self.effect_estimator.compute_uncertainty(contexts, self.n_samples)
        return variances

    def remember(self, timestep: Timestep):
        self.memory.append(timestep)


    
    