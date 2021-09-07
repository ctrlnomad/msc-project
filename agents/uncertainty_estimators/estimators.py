
from typing import Callable, List

import torch


import torch.optim as optim
from torch.utils.data import DataLoader

from agents.uncertainty_estimators.arches import train_loop, cate_train_loop

class BaseEstimator:
    def __init__(self, make: Callable, config, deconfound_fn: Callable) -> None: # potentially a summary writer, shall pass inconfig?
        self.config = config
        self.deconfound_fn = deconfound_fn

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



class DropoutEstimator(BaseEstimator):
    def __init__(self, make: Callable, config, deconfound_fn: Callable = None, CATE=False) -> None:
        super().__init__(make, config, deconfound_fn)
        self.net = make(self.config) 

        if self.config.cuda:
            self.net = self.net.cuda()

        self.opt = optim.Adam(self.net.parameters(), lr=self.config.lr)
        self.train_fn = cate_train_loop if CATE else train_loop

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        # compute entropy with dropout
        bs = len(contexts)
        result = torch.zeros((2, bs, self.config.mc_samples))
        if self.config.cuda:
            result = result.cuda()
            contexts = contexts.cuda()

        for i in range(self.config.mc_samples):
            effects = self.net(contexts)[0]
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
            effects = self.net(contexts)[0]
            effects = torch.stack(effects).squeeze()
            if bs ==1: 
                effects = effects[None].T
            result[..., i] = effects[1, :] - effects[0, :] 
        return result.var(dim=-1)

    def train(self, loader):
        return self.train_fn(self.net, loader, self.opt, self.config, deconfound=self.deconfound_fn)

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        mu, sigma = self.net(contexts)
        self.net.train()
        return torch.stack(mu).squeeze(), torch.stack(sigma).squeeze()


    @property
    def models(self):
        return self.net


class EnsembleEstimator(BaseEstimator):
    def __init__(self, make: Callable, config, deconfound_fn: Callable = None, CATE = False) -> None: 
        super().__init__(make, config, deconfound_fn)
        self.ensemble = [make(config) for _ in range(self.config.ensemble_size)]

        if self.config.cuda:
            self.ensemble = [n.cuda() for n in self.ensemble]

        self.train_fn = cate_train_loop if CATE else train_loop
        self.opt_cls = optim.Adam
        self.opts = [self.opt_cls(n.parameters(), lr=self.config.lr) for n in self.ensemble]
        

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        contexts = contexts.cuda() if self.config.cuda else contexts
        results = [torch.stack(net(contexts)[0]) for net in self.ensemble] 

        results = torch.stack(results).squeeze().var(dim=0)
        return results


    def compute_cate_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        contexts = contexts.cuda() if self.config.cuda else contexts
        diff = lambda treat, no_treat: treat - no_treat

        results = [net(contexts) for net in self.ensemble]
        results = [mu[1] - mu[0] for mu in results]

        results = torch.stack(results).squeeze().var(dim=0)
        return results

    def train(self, loader: DataLoader)  -> List[List[float]]: 
        losses = [[] for _ in range(self.config.ensemble_size)]
        
        for i in range(self.config.ensemble_size):
            losses[i].append(self.train_fn(self.ensemble[i], loader, self.opts[i], self.config, deconfound=self.deconfound_fn))
        
        return losses

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        for m in self.ensemble: m.eval()
        outputs = [net(contexts) for net in self.ensemble]

        mus = [torch.stack(output[0]) for output in outputs]
        mus = torch.stack(mus).squeeze()
        mu = mus.mean(dim=0)
        
        sigmas = [torch.stack(output[1]) for output in outputs]
        sigmas = torch.stack(sigmas).squeeze()
        sigma =  sigmas.mean(dim=0) + mu.var(dim=0) # total variance
        for m in self.ensemble: m.train()
        return mu, sigma

    @property
    def models(self):
        return self.ensemble