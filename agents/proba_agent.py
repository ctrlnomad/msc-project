from typing import Tuple, List
from dataclasses import dataclass, field
import numpy as np
from collections import deque
from numpy.core.fromnumeric import nonzero

import torch
import torch.nn as nn
from torch.nn.modules import utils
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.distributions as distributions

from causal_env.envs import Timestep, TimestepDataset
from agents.base_agent import BaseAgent

import utils
import utils.mnist 

import logging
logger = logging.getLogger(__name__)


def mlp(inp):
    return nn.Sequential(
            nn.Linear(inp, 50),
            nn.LeakyReLU(),
            nn.Dropout(p=1/4),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.Dropout(p=1/4)
        )

class BayesianConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
        )

        self.net = mlp(320)

        self.mu_theta = nn.Linear(25, 2)
        self.sigma_theta = nn.Sequential(
            nn.Linear(25, 2), 
            nn.ReLU()
        ) # this is a bit clapped, ask in the meeting tomorrow TODO

    def forward(self, x, add_delta=True):
        x = self.conv_block(x)
        x = x.view(-1, 320)

        emb  = self.net(x)
        mu_pred = self.mu_theta(emb)
        sigma_pred = self.sigma_theta(emb)

        if add_delta:
            sigma_pred += 1e-3 # avoid numerical errors

        return mu_pred, sigma_pred
        

    def compute_uncertainty(self, x,  n_samples):
        # compute entropy with dropout
        bs = len(x)
        result = torch.zeros((bs, 2, n_samples)) # two for binary treatment
        for i in range(n_samples):
            result[..., i] = self(x)[0]
        return result.var(dim=-1) # Var[E[Y | X]]

@dataclass
class VariationalAgentConfig:
    dim_in: Tuple[int] = (1, 28, 28)
    memsize: int = 100_000
    mc_samples: int = 100

    do_nothing: float = 0.5 # do nothing for this proportion of steps
    cuda: bool = False
    batch_size:int = 32


@dataclass
class History:
    loss: List[float] = field(default_factory=list)


class VariationalAgent(BaseAgent):
    """
    half the round we learn by empty interventions
    the other times we learn by choosing based on epsitemic uncertainty
    we evaluate by regret. 
    """
    def __init__(self, config: VariationalAgentConfig):
        # learning is the ITE of causal arms
        self.memory = deque(maxlen=config.memsize)

        self.effect_estimator = BayesianConvNet()
        self.config = config

        self.history = History()
        self.digit_sampler = utils.mnist.MnistSampler()
        
        if self.config.cuda:
            self.effect_estimator.cuda()

    def observe(self, timestep: Timestep):
        self.memory.append(timestep)

    def train(self, n_epochs:int=1):
        if len(self.memory) <= self.config.batch_size:
            logger.info('agent not training, not enough data')
            return 

        for e in range(n_epochs):
            logger.info(f'[{e}] starting training ...')
            self.train_once()
            logger.info(f'[{e}] training finished; loss is at: [{self.history.loss[-1]:.4f}]')
        
    def act(self, timestep: Timestep):
        uncertaitnties = self.compute_digit_uncertainties(timestep.context)
        loc = (uncertaitnties.max() == uncertaitnties).nonzero().squeeze()
        intervention = loc[0] + len(timestep.context)* loc[1]
        return intervention.item()
    
    
    def train_once(self, ):
        dataset = TimestepDataset(self.memory)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        opt = optim.Adam(self.effect_estimator.parameters(),lr=3e-4)

        for contexts, treatments, effects in loader:
            opt.zero_grad()
            
            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments
            
            # contextsss reshape bbbbbb
            contexts =contexts.view(-1, *self.config.dim_in)
            treatments = treatments.flatten()
            effects = effects.repeat(10)
            mu_pred, sigma_pred = self.effect_estimator(contexts, add_delta=True)

            mu_pred = mu_pred.gather(0, treatments[None].T)
            sigma_pred = sigma_pred.gather(0, treatments[None].T)

            sigma_mat_pred = utils.to_diag_var(sigma_pred)
            loss = -distributions.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()
            loss.backward()
            opt.step()

            self.history.loss.append(loss.item()) # better way of doing this tho

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        variances = self.effect_estimator.compute_uncertainty(contexts, self.config.mc_samples)
        return variances 

    def compute_digit_distributions(self, contexts: torch.Tensor):
        return self.effect_estimator(contexts)

    def compute_best_action(self, contexts: torch.Tensor):
        mu, _ = self.effect_estimator(contexts)
        return (mu.max() == mu).nonzero().squeeze()