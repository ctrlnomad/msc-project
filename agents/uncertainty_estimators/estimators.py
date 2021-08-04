
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
        self.opt = optim.Adam(self.net.parameters())

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        # compute entropy with dropout
        bs = len(contexts)
        result = torch.zeros((bs, 2, self.config.mc_samples)) 
        print(result.shape)
        for i in range(self.config.mc_samples):
            effects = self.net(contexts)[0]
            effects = torch.stack(effects).squeeze().T
            result[..., i] = effects
        return result.var(dim=-1) # Var[E[Y | X]]

    def train(self, loader):
        return train_loop(self.net, loader, self.opt, self.config.cuda)

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.net(contexts)




class EnsembleEstimator(BaseEstimator):
    def __init__(self, make: Callable, config ) -> None: # TODO maybe set dropout to 0; check docs
        super().__init__(make, config)
        self.ensemble = [make(config) for _ in range(self.config.ensemble_size)]
        self.opt_cls = optim.Adam
        self.opts = [self.opt_cls(n.parameters()) for n in self.ensemble]
        

    def compute_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        results = [torch.stack(net(contexts)[0]) for net in self.ensemble] 
        results = torch.stack(results)

        return results.var(dim=-1)

    def train(self, loader: DataLoader)  -> List[List[float]]: 
        losses = [[] for _ in range(self.config.ensemble_size)]
        
        for i in range(self.config.ensemble_size):
            losses[i].append(train_loop(self.ensemble[i], loader, self.opts[i], self.config.cuda))
        
        return losses

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        results = [torch.stack(net(contexts)[0]) for net in self.ensemble] 
        results = torch.stack(results)

        return results.mean(dim=-1)