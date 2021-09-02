from typing import Callable, List

import torch

import torch.distributions as D
from torch.utils import data
import utils

import torch.optim as optim
from torch.utils.data import DataLoader

from agents.effect.estimators import EffectEstimator



class BaseUncertainty:

    def compute_uncertainty(self, contexts: torch.Tensor): # -> compute ite-uncertainty
        raise NotImplementedError()

    def train(self, loader: DataLoader) -> List[float]: 
        raise NotImplementedError()

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute_cate_uncertainty(self, contexts: torch.Tensor): # -> compute ite-uncertainty
        raise NotImplementedError()

    @property
    def models(self):
        raise NotImplementedError()



class DropoutEstimator(BaseUncertainty):
    def __init__(self, config) -> None:
        self.config = config
        self.estimator = EffectEstimator(self.config) 


    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        # compute entropy with dropout
        bs = len(contexts)
        result = torch.zeros((2, bs, self.config.mc_samples))
        if self.config.cuda:
            result = result.cuda()
            contexts = contexts.cuda()

        for i in range(self.config.mc_samples):
            effects = self.estimator(contexts)[0]
            effects = torch.stack(effects).squeeze()
            if bs ==1: # hack
                effects = effects[None].T
            result[..., i] = effects
        return result.var(dim=-1) # Var[E[Y | X]]

    def compute_cate_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        # compute entropy with dropout
        bs = len(contexts)
        result = torch.zeros((bs, self.config.mc_samples))
        if self.config.cuda:
            result = result.cuda()
            contexts = contexts.cuda()

        for i in range(self.config.mc_samples):
            effects = self.estimator(contexts)[0]
            effects = torch.stack(effects).squeeze()
            if bs ==1: 
                effects = effects[None].T
            result[..., i] = effects[1, :] - effects[0, :] 
        return result.var(dim=-1)

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.estimator(contexts)

    def train(self, dataset,  num_epochs=2):
        self.estimator.train(dataset,  num_epochs=num_epochs)

    @property
    def models(self):
        return self.net


class EnsembleEstimator(BaseUncertainty):
    def __init__(self, config) -> None: 
        self.config = config
        self.ensemble = [EffectEstimator(config) for _ in range(self.config.ensemble_size)]

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        contexts = contexts.cuda() if self.config.cuda else contexts
        results = [torch.stack(estimator(contexts)[0]) for estimator in self.ensemble] 

        results = torch.stack(results).squeeze().var(dim=0)
        return results


    def compute_cate_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        contexts = contexts.cuda() if self.config.cuda else contexts

        results = [net(contexts) for net in self.ensemble]
        results = [mu[1] - mu[0] for mu in results]

        results = torch.stack(results).squeeze().var(dim=0)
        return results

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        outputs = [estimator(contexts) for estimator in self.ensemble]

        mus = [torch.stack(output[0]) for output in outputs]
        mus = torch.stack(mus).squeeze()
        mu = mus.mean(dim=0)
        
        sigmas = [torch.stack(output[1]) for output in outputs]
        sigmas = torch.stack(sigmas).squeeze()
        sigma =  sigmas.mean(dim=0) + mu.var(dim=0) # total variance
        return mu, sigma


    def train(self, dataset, num_epochs=2):
        for e in self.ensemble:
            e.train(dataset, num_epochs=num_epochs)

    @property
    def models(self):
        return self.ensemble
