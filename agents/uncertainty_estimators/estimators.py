
from typing import Callable, List

import torch


import torch.optim as optim
from torch.utils.data import DataLoader

from agents.uncertainty_estimators.arches import train_loop

class BaseEstimator:
    def __init__(self, make: Callable, config) -> None: # potentially a summary writer, shall pass inconfig?
        self.config = config

    def compute_uncertainty(self, contexts: torch.Tensor):
        raise NotImplementedError()

    def train(self, loader: DataLoader) -> List[float]: 
        raise NotImplementedError()

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()




class DropoutEstimator(BaseEstimator):
    def __init__(self, make: Callable, config) -> None:
        super().__init__(make, config)
        self.net = make(self.config) 

        if self.config.cuda:
            self.net = self.net.cuda()

        self.opt = optim.Adam(self.net.parameters())

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        # compute entropy with dropout
        bs = len(contexts)
        result = torch.zeros((2, bs, self.config.mc_samples))
        if self.config.cuda:
            result = result.cuda()
            contexts = contexts.cuda()

        for i in range(self.config.mc_samples):
            effects = self.net(contexts)[0]
            effects = torch.stack(effects).squeeze().T
            result[..., i] = effects
        return result.var(dim=-1) # Var[E[Y | X]]

    def train(self, loader):
        return train_loop(self.net, loader, self.opt, self.config)

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.net(contexts)
        return torch.stack(mu).squeeze(), torch.stack(sigma).squeeze()




class EnsembleEstimator(BaseEstimator):
    def __init__(self, make: Callable, config ) -> None: 
        super().__init__(make, config)
        config.dropout_rate = 0 # TODO maybe set dropout to 0; check docs
        self.ensemble = [make(config) for _ in range(self.config.ensemble_size)]

        if self.config.cuda:
            self.ensemble = [n.cuda() for n in self.ensemble]

        self.opt_cls = optim.Adam
        self.opts = [self.opt_cls(n.parameters()) for n in self.ensemble]
        

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        contexts = contexts.cuda() if self.config.cuda else contexts
        results = [torch.stack(net(contexts)[0]) for net in self.ensemble] 

        results = torch.stack(results).squeeze().var(dim=0)
        return results

    def train(self, loader: DataLoader)  -> List[List[float]]: 
        losses = [[] for _ in range(self.config.ensemble_size)]
        
        for i in range(self.config.ensemble_size):
            losses[i].append(train_loop(self.ensemble[i], loader, self.opts[i], self.config))
        
        return losses

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        outputs = [net(contexts) for net in self.ensemble]

        mu = [torch.stack(output[0]) for output in outputs]
        mu = torch.stack(mu).squeeze().mean(dim=0)
        
        sigma = [torch.stack(output[1]) for output in outputs]
        sigma = torch.stack(sigma).squeeze().mean(dim=0)
        return mu, sigma