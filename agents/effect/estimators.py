
from typing import Callable, List

import torch

import torch.distributions as D
from torch.utils import data
import utils

import torch.optim as optim
from torch.utils.data import DataLoader

from agents.effect.arches import EffectNet, StructNet



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


class EffectEstimator:
    def __init__(self, config, causal_model: torch.nn.Module) -> None:
        self.config = config
        self.net = EffectNet(config)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.opt = optim.Adam(self.net.parameters())

        self.causal_model = causal_model

    def train(self, dataset, num_epochs=2):
        for _ in range(num_epochs):
            self.train_once(dataset)

    def train_once(self, dataset):

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for contexts, treatments, true_causal, effects in loader:
                
            contexts = contexts.cuda() if self.config.cuda else contexts
            effects = effects.cuda() if self.config.cuda else effects
            treatments = treatments.cuda() if self.config.cuda else treatments

            if self.causal_model:
                effects = self.deconfound(contexts, treatments, effects, true_causal)
            else:
                effects = torch.repeat_interleave(effects, self.config.num_arms, 1)

            effects = effects.flatten()
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

            loss = -D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()
            
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

    def deconfound(self,  contexts, treatments, effects, true_causal):
        # now the idea is different, contexts are scaled to the degree of their causality
        # freeze(effect_net)
        bs = len(contexts)
        
        contexts = contexts.clone().detach()
        flat_contexts = contexts.view(-1, *self.config.dim_in)
        
        causal_ids = self.causal_model(true_causal)
        causal_ids = causal_ids.unsqueeze(-1).repeat([1, 1, 28, 28])

        flat_contexts = flat_contexts * causal_ids

        with torch.no_grad():
            mu_pred, _ = self.net(flat_contexts)

        mu_pred = torch.stack(mu_pred)
        mu_pred = mu_pred.view(2, *contexts.shape[:2])

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
        self.net.eval()
        effects = self.net(contexts)
        self.net.train()
        return effects