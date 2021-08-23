
from typing import Callable, List

import torch

import torch.distributions as D
import utils

import torch.optim as optim
from torch.utils.data import DataLoader

from agents.uncertainty_estimators.arches import ConvNet, train_loop, cate_train_loop

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

        self.opt = optim.Adam(self.net.parameters())
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
        self.opts = [self.opt_cls(n.parameters()) for n in self.ensemble]
        

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



class MetaCausalEstimator(BaseEstimator):
    def __init__(self, config) -> None:
        super().__init__(None, config, None)

        self.net = ConvNet(config, causal_model=True)
        if self.config.cuda:
            self.net = self.net.cuda()
        self.opt = optim.Adam(self.net.parameters())

    def compute_beliefs(self, contexts, return_proba=False): # E-step
        flat_contexts = contexts.view(-1, *self.config.dim_in) #  TODO? does this work
        _, _, causal_ids = self.net(flat_contexts)
        if return_proba:
            return causal_ids 
        causal_context_ids = (causal_ids > 0.5).squeeze().view(*contexts.shape[:2])

        return causal_context_ids

        
    def train(self, loader): # put training loop inside here?

        for contexts, treatments, _, effects in loader:
            self.opt.zero_grad()
            
            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments

            causal_model = self.compute_beliefs(contexts)
            effects = self.deconfound(causal_model, contexts, treatments, effects)
            effects = effects.flatten()

            contexts = contexts.view(-1, *self.config.dim_in)
            treatments = treatments.flatten()

            mu_pred, sigma_pred, _ = self.net(contexts)

            mu_pred = torch.stack(mu_pred).squeeze()
            sigma_pred = torch.stack(sigma_pred).squeeze()

            mu_pred = mu_pred.gather(0, treatments[None]) # 2, num_arms
            sigma_pred = sigma_pred.gather(0, treatments[None])

            if self.config.fixed_sigma:
                sigma_pred = torch.ones_like(sigma_pred) * 0.1

            sigma_mat_pred = utils.to_diag_var(sigma_pred, cuda=self.config.cuda)

            loss = -D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()

            loss.backward()
            self.opt.step()

        return loss.item()


    def deconfound(self, causal_model, contexts, treatments, effects):
        with torch.no_grad():
            bs = len(contexts)

            causal_idxs = torch.nonzero(causal_model, as_tuple=False)
            causal_treatments = torch.masked_select(treatments, causal_model)

            if causal_idxs.nelement() == 0:
                effects = torch.zeros(bs, self.config.num_arms)
                return effects.cuda() if self.config.cuda else effects

            confounding_contexts = torch.stack([contexts[bidx, sidx] for bidx, sidx in causal_idxs])
            mu_pred = self(confounding_contexts)[0]

            deconfounded_effects  = torch.zeros(bs, self.config.num_arms)

            if self.config.cuda:
                deconfounded_effects = deconfounded_effects.cuda()

            if len(mu_pred.shape) != len(causal_treatments[None].shape):
                mu_pred = mu_pred.unsqueeze(1)

            mu_pred = mu_pred.gather(0, causal_treatments[None]).squeeze()

            for i in range(bs):
                total_batch_effect = torch.masked_select(mu_pred ,causal_idxs[:, 0] == i)
                deconfounded_effects[i, causal_model[i]] = effects[i] - (total_batch_effect.sum() - total_batch_effect)

            return deconfounded_effects

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        mu, sigma, proba = self.net(contexts)
        self.net.train()
        return torch.stack(mu).squeeze(), torch.stack(sigma).squeeze(), proba


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

    @property
    def models(self):
        return self.net
