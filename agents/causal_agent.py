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
import utils
import utils.mnist as mnist

import logging
logger = logging.getLogger(__name__)




@dataclass
class CausalAgentConfig:
    dim_in: Tuple[int] = (1, 28, 28)
    memsize: int = 100_000
    mc_samples: int = 100
    ensemble_size: int =100

    do_nothing: float = 0.5 
    cuda: bool = False
    batch_size:int =32



@dataclass
class History:
    loss: List[float] = field(default_factory=list)


class CausalAgent:

    def __init__(self, config: CausalAgentConfig, causal_model: List[int]):
        # learning is the ITE of causal arms
        self.memory = deque(maxlen=config.memsize)

        self.effect_estimator = BayesianConvNet()
        self.config = config

        self.history = History()

        self.causal_model = torch.Tensor(causal_model).long()
        self.causal_ids = self.causal_model.nonzero().squeeze()

        if self.config.cuda:
            self.effect_estimator.cuda()

        self.mnist_sampler = mnist.MnistSampler()

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
        loc = (uncertaitnties.max() == uncertaitnties).nonzero()
        intervention = loc[0] + len(timestep.context)* loc[1]
        return intervention.item()
    
    

    def decounfound(self, contexts, treatments,  effects):
        with torch.no_grad():
            causal_treatments = treatments.index_select(dim=1, index=self.causal_ids)
            bs = len(contexts)

            confounding_contexts = contexts.index_select(dim=1, index=self.causal_ids).view(-1, *self.config.dim_in)
            mu_pred, _ = self.effect_estimator(confounding_contexts)
            mu_pred = mu_pred.view(bs, -1, 2)
            # we know how many causal effects there are 4 in this case
            num_causal_arms = len(self.causal_ids)
            deconfounded_effects  = torch.zeros(bs, num_causal_arms)

            mu_pred = mu_pred.gather(-1, causal_treatments[...,None]).squeeze()
            for i in range(bs):
                deconfounded_effects[i] = effects[i] - (mu_pred[i].sum() - mu_pred[i])

            deconfounded_effects = deconfounded_effects.flatten()
            causal_treatments = causal_treatments.flatten()

            return confounding_contexts,causal_treatments, deconfounded_effects



    def train_once(self):
        dataset = TimestepDataset(self.memory)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        opt = optim.Adam(self.effect_estimator.parameters(),lr=3e-4)

        for contexts, treatments, effects in loader:
            opt.zero_grad()
            
            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments
            
            contexts, treatments, effects = self.decounfound(contexts, treatments, effects)
            mu_pred, sigma_pred = self.effect_estimator(contexts, add_delta=True)

            mu_pred = mu_pred.gather(0, treatments[None].T)
            sigma_pred = sigma_pred.gather(0, treatments[None].T)

            sigma_mat_pred = utils.to_diag_var(sigma_pred)
            loss = -distributions.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()
            loss.backward()
            opt.step()

            self.history.loss.append(loss.item()) 

    def compute_digit_uncertainties(self, contexts: torch.Tensor) -> np.ndarray:
        variances = self.effect_estimator.compute_uncertainty(contexts, self.config.mc_samples)
        return variances.detach().cpu().numpy()

    def compute_digit_distributions(self, contexts: torch.Tensor)-> np.ndarray:
        mu, sigma = self.effect_estimator(contexts)
        mu = mu.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
        return mu, sigma

    def compute_best_action(self, contexts: torch.Tensor)-> np.ndarray:
        mu, _ = self.effect_estimator(contexts)
        return (mu.max() == mu).nonzero().squeeze().detach().cpu().numpy()

