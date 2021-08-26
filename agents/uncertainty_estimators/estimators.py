
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

        
    def train(self, dataset, num_epochs=2):

        contexts, treatments, _, effects = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=True)))
        
        self.net.proc_dataset(contexts, treatments, effects)
        
        # freeze the struct model?
        # inner loop, freeze?

        self.effect_estimator = EffectEstimator(self.config, self.net)
        self.effect_estimator.train(dataset, num_epochs=num_epochs, use_causal=True) # a new effect estimator trained to convergence 
        nll = self.effect_estimator.compute_nll(contexts, treatments, effects) # compute LL with the entire dataset

        self.effect_estimator.train(dataset, num_epochs=num_epochs, use_zero=True) # a new effect estimator trained to convergence 
        nll = self.effect_estimator.compute_nll(contexts, treatments, effects) # compute LL with the entire dataset

        # outer loop        
        contexts, treatments, _, effects = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=True)))
        nll = self.effect_estimator.compute_nll(contexts, treatments, effects) # compute LL with the entire dataset

        if self.config.cuda:
            contexts = contexts.cuda()

        flat_contexts = contexts.view(-1, *self.config.dim_in)
        causal_probabilities = self.net(flat_contexts)

        post_prob = causal_probabilities
        prior_prob = torch.ones_like(causal_probabilities) * self.config.prior

        if self.config.cuda:
            post_prob = post_prob.cuda()
            prior_prob = prior_prob.cuda()

        post = D.Bernoulli(post_prob)
        prior = D.Bernoulli(prior_prob)

        kl = D.kl_divergence(post, prior).mean()

        loss = nll + kl

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), nll.item(), kl.item()

    @property
    def models(self):
        return [self.net, self.effect_estimator.net]

    def compute_beliefs(self, contexts):
        if hasattr(self.net, 'dataset_emb'):
            probas = self.net(contexts)
        else:
            probas = torch.zeros(self.config.num_arms)
        return probas


class EffectEstimator(DropoutEstimator):
    def __init__(self, config, causal_model: torch.nn.Module) -> None:
        self.config = config
        self.net = EffectNet(config)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.opt = optim.Adam(self.net.parameters())

        self.causal_model = causal_model

    def train(self, dataset, num_epochs=2, use_causal=False, use_zero=False):
        for _ in range(num_epochs):
            self.train_once(dataset, use_causal=use_causal, use_zero=use_zero)

    def train_once(self, dataset, use_causal=False, use_zero=False):

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for contexts, treatments, true_causal, effects in loader:
            
                
            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments

            effects2 = self.deconfound(contexts, treatments, effects, true_causal=true_causal if use_causal else None)

            effects3 = effects2.flatten()
            contexts = contexts.view(-1, *self.config.dim_in)
            treatments = treatments.flatten()

            mu_pred, sigma_pred = self.net(contexts)

            mu_pred = torch.stack(mu_pred).squeeze()
            sigma_pred = torch.stack(sigma_pred).squeeze()

            mu_pred = mu_pred.gather(0, treatments[None])
            sigma_pred = sigma_pred.gather(0, treatments[None])

            if self.config.fixed_sigma:
                sigma_pred = torch.ones_like(sigma_pred) * 0.1

            sigma_mat_pred = utils.to_diag_var(sigma_pred, cuda=self.config.cuda)

            loss = -D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects3).mean()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return loss

    def compute_nll(self, contexts, treatments, effects):

        contexts = contexts.cuda() if self.config.cuda else contexts
        effects = effects.cuda() if self.config.cuda else effects
        treatments = treatments.cuda() if self.config.cuda else treatments

        effects2 = self.deconfound(contexts, treatments, effects)

        effects3 = effects2.flatten()
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

        nll = -D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects3).mean()

        return nll

    def deconfound(self,  contexts, treatments, effects, true_causal=None):
        # now the idea is different, contexts are scaled to the degree of their causality
        # freeze(effect_net)

        bs = len(contexts)
        
        contexts2 = contexts.clone().detach()
        flat_contexts = contexts2.view(-1, *self.config.dim_in)
        causal_probas = self.causal_model(flat_contexts)

        # causal_probas = causal_probas.view(*contexts.shape[:2])
        dist = D.RelaxedBernoulli(1, probs=causal_probas) # should we sample or can we use directly? TODO

        causal_probas_sample = dist.rsample() 

        causal_sample2 = causal_probas_sample.unsqueeze(-1).repeat([1, 1, 28, 28])

        flat_contexts2 = flat_contexts * causal_sample2

        with torch.no_grad():
            mu_pred, _ = self.net(flat_contexts2)

        mu_pred = torch.stack(mu_pred)
        mu_pred = mu_pred.view(2, *contexts.shape[:2]) # here we need this to deconfound-timestep

        deconfounded_effects = torch.zeros(bs, self.config.num_arms)

        if self.config.cuda:
            deconfounded_effects = deconfounded_effects.cuda()

        mu_pred = mu_pred.gather(0, treatments[None]).squeeze()

        for i in range(bs):
            deconfounded_effects[i] = effects[i] - (mu_pred[i].sum() - mu_pred[i])

        return deconfounded_effects

    def __call__(self, contexts: torch.Tensor) -> torch.Tensor:
        if self.config.cuda:
            contexts = contexts.cuda()
        return self.net(contexts)