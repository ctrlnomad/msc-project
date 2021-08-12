import torch
import torch.nn as nn
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.nn.functional as F

import torchvision.models as models
import utils

from typing import Callable, List, Tuple

import torch.optim as optim
from torch.utils.data import DataLoader

class EffectNet(nn.Module):
    def __init__(self, inp: int, dropout_rate=1/2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.emb = nn.Sequential(
            nn.Linear(inp, 25),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate))
            
        self.mu = nn.Linear(25, 1)
        self.sigma = nn.Linear(25, 1)

    def forward(self, x):
        emb = self.emb(x)
        return self.mu(emb),  F.relu(self.sigma(emb))

class ConvNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout_rate = config.dropout_rate
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=self.dropout_rate),
            nn.Flatten()
        )
        
        self.treat = EffectNet(320, dropout_rate=self.dropout_rate)
        self.no_treat = EffectNet(320, dropout_rate=self.dropout_rate)

    def forward(self, x, add_delta=True):
        emb = self.conv_block(x)

        treat_mu, treat_sigma = self.treat(emb)
        no_treat_mu, no_treat_sigma = self.no_treat(emb)

        if add_delta:
            treat_sigma += 1e-7
            no_treat_sigma += 1e-7

        return (treat_mu, no_treat_mu), (treat_sigma, no_treat_sigma)


    def __repr__(self):
        return "ConvNet"
        

class ResNet(nn.Module): 
    def __init__(self, config) -> None:
        self.config = config
        self.resnet = models.resnet18(pretrained=config.pretrained)

        # need a transform here, cause input is not 28 by 28
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, 320) 

        self.treat = EffectNet(320)
        self.no_treat = EffectNet(320)

    def forward(self, x, add_delta=True):
        emb = self.resnet(x)

        treat_mu, treat_sigma = self.treat(emb)
        no_treat_mu, no_treat_sigma = self.no_treat(emb)

        if add_delta:
            treat_sigma += 1e-7
            no_treat_sigma += 1e-7

        return (treat_mu, no_treat_mu), (treat_sigma, no_treat_sigma)


    def __repr__(self):
        return "ResNet18"



def train_loop(model:nn.Module, loader: DataLoader, opt: optim.Optimizer, \
            config, deconfound: Callable = None) -> List[float]: 

    losses = []

    for contexts, treatments, causal_ids, effects in loader:
        opt.zero_grad()
        
        contexts = contexts.cuda() if config.cuda else contexts
        effects = effects.cuda() if config.cuda else effects
        treatments = treatments.cuda() if config.cuda else treatments
        causal_ids = causal_ids.cuda() if config.cuda else causal_ids


        if deconfound:
            effects = deconfound(contexts, treatments, causal_ids, effects)
            effects = effects.flatten()
        else:
            effects = torch.repeat_interleave(effects, config.num_arms)
            # 4,5,6 -> [-1, -1, -1]

        contexts = contexts.view(-1, *config.dim_in)
        treatments = treatments.flatten()

        mu_pred, sigma_pred = model(contexts, add_delta=True)

        mu_pred = torch.stack(mu_pred).squeeze()
        sigma_pred = torch.stack(sigma_pred).squeeze()

        mu_pred = mu_pred.gather(0, treatments[None]) # 2, num_arms
        sigma_pred = sigma_pred.gather(0, treatments[None])

        sigma_mat_pred = utils.to_diag_var(sigma_pred, cuda=config.cuda)
        
        # TODO: Mixture
        # distributions.Categorical(torch.)
        # loss = -D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()
        mix =  D.Categorical(torch.ones(len(effects)) / len(effects))
        comp = D.Normal(mu_pred, sigma_pred)

        gmm = MixtureSameFamily(mix, comp)
        loss = -gmm.log_prob(effects).mean()

        loss.backward()
        opt.step()
        # TODO: thought experiment

        losses.append(loss.item()) # better way of doing this tho
    
    return losses

