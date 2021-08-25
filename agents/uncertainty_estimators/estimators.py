
from typing import Callable, List

import torch

import torch.distributions as D
from torch.utils import data
import utils

import torch.optim as optim
from torch.utils.data import DataLoader

from agents.uncertainty_estimators.arches import EffectNet, StructNet, train_loop

class BaseEstimator:

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
        self.make, self.config, self.deconfound_fn = make, config, deconfound_fn
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
        self.make, self.config, self.deconfound_fn = make, config, deconfound_fn
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


    def compute_cate_uncertainty(self, contexts: torch.Tensor) -> torch.Tensor:
        contexts = contexts.cuda() if self.config.cuda else contexts
        diff = lambda treat, no_treat: treat - no_treat

        results = [net(contexts) for net in self.ensemble]
        results = [mu[1] - mu[0] for mu in results]

        results = torch.stack(results).squeeze().var(dim=0)
        return results

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
class StructEstimator:
    def __init__(self, config) -> None:
        self.config = config
        self.net = StructNet(config)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.opt = optim.Adam(self.net.parameters())

        self.effect_estimator = EffectEstimator(config, self.net)

        
    def train(self, dataset):
        self.opt.zero_grad()

        contexts, treatments, _, effects = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=True)))
        
        self.net.proc_dataset(contexts, treatments, effects)
        
        # freeze the struct model of smt ?
        self.effect_estimator.train(dataset)
        
        contexts, treatments, _, effects = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=True)))
        ll = self.effect_estimator.compute_ll(contexts, treatments, effects) # with the trained thing


        flat_contexts = contexts.view(-1, *self.config.dim_in)
        causal_probabilities = self.net(flat_contexts)

        post = D.Bernoulli(causal_probabilities)
        prior = D.Bernoulli(torch.ones_like(causal_probabilities) * 1e-4)

        kl = D.kl_divergence(post, prior).sum()


        loss = ll + kl
        loss.backward()

        self.opt.step()

    @property
    def models(self):
        return [self.net, self.effect_estimator.net]



class EffectEstimator(DropoutEstimator):
    def __init__(self, config, causal_model: torch.nn.Module) -> None:
        self.config = config
        self.net = EffectNet(config)
        if self.config.cuda:
            self.net = self.net.cuda()
        self.opt = optim.Adam(self.net.parameters())

        self.causal_model = causal_model

    def train(self, dataset):

        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        for contexts, treatments, _, effects in loader:

            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments

            effects = self.deconfound(contexts, treatments, effects)

            effects = effects.flatten()
            contexts = contexts.view(-1, *self.config.dim_in)
            treatments = treatments.flatten()

            mu_pred, sigma_pred = self.net(contexts)

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

        return loss

    def compute_ll(self, contexts, treatments, effects):

        contexts = contexts.cuda() if self.config.cuda else contexts
        effects = effects.cuda() if self.config.cuda else effects
        treatments = treatments.cuda() if self.config.cuda else treatments

        effects = self.deconfound(contexts, treatments, effects)

        effects = effects.flatten()
        contexts = contexts.view(-1, *self.config.dim_in)
        treatments = treatments.flatten()

        self.net.eval()
        mu_pred, sigma_pred = self.net(contexts)
        self.net.train()

        mu_pred = torch.stack(mu_pred).squeeze()
        sigma_pred = torch.stack(sigma_pred).squeeze()

        mu_pred = mu_pred.gather(0, treatments[None]) # 2, num_arms
        sigma_pred = sigma_pred.gather(0, treatments[None])

        if self.config.fixed_sigma:
            sigma_pred = torch.ones_like(sigma_pred) * 0.1

        sigma_mat_pred = utils.to_diag_var(sigma_pred, cuda=self.config.cuda)

        ll = D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()

        return ll

    def deconfound(self,  contexts, treatments, effects):
        # now the idea is different, contexts are scaled to the degree of their causality
        # freeze(effect_net) ?
        with torch.no_grad():
            bs = len(contexts)
            flat_contexts = contexts.view(-1, *self.config.dim_in)
            causal_probas = self.causal_model(flat_contexts)

            # causal_probas = causal_probas.view(*contexts.shape[:2])
            dist = D.RelaxedBernoulli(1, probs=causal_probas) # should we sample or can we use directly?

            causal_probas_sample = dist.rsample()
            
            for i in range(flat_contexts.shape[0]): # i dont know if there is a better way
                flat_contexts[i] *= causal_probas_sample[i]

            mu_pred, _ = self.net(flat_contexts)
            mu_pred = torch.stack(mu_pred)
            mu_pred = mu_pred.view(2, *contexts.shape[:2]) # here we need this to deconfound-timestep

            deconfounded_effects = torch.zeros(bs, self.config.num_arms)

            if self.config.cuda:
                deconfounded_effects = deconfounded_effects.cuda()

            mu_pred = mu_pred.gather(0, treatments[None]).squeeze()

            for i in range(bs):
                deconfounded_effects[i] = effects[i] - (mu_pred[i].sum() - mu_pred[i])

        return deconfounded_effects