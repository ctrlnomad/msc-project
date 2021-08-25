import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.utils import data

import torchvision.models as models
import utils

from typing import Callable, List, Tuple

import torch.optim as optim
from torch.utils.data import DataLoader



class EffectNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout_rate = config.dropout_rate
        self.conv_block = ConvBlock(config)

        self.emb = nn.Sequential(
            nn.Linear(320, 25),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate))
            
        
        self.treat_mu = nn.Linear(25, 1)
        self.treat_sigma = nn.Linear(25, 1)

        self.no_treat_mu = nn.Linear(25, 1)
        self.no_treat_sigma = nn.Linear(25, 1)

    def forward(self, x):
        x = self.conv_block(x)
        emb = self.emb(x)

        treat_mu = self.treat_mu(emb)
        treat_sigma = 1e-7 + torch.sigmoid(self.treat_sigma(emb))
        
        no_treat_mu = self.no_treat_mu(emb)
        no_treat_sigma = 1e-7 + torch.sigmoid(self.treat_sigma(emb))

        return (no_treat_mu, treat_mu ), (treat_sigma, no_treat_sigma)


class ConvBlock(nn.Module):

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

    def forward(self, x, merge=False):
        if merge:
            shape = x.shape
            x = x.view(-1, *shape[2:])

        return self.conv_block(x)


    def __repr__(self):
        return "ConvNet"

        
class AttentionEncoder(nn.Module): # what is this
  def __init__(self, dims, hidden):
    super().__init__()

    self.attention = torch.nn.MultiheadAttention(hidden, 1)
    
    self.key = nn.Linear(dims, hidden)
    self.query = nn.Linear(320, hidden) # conv embedding dimension
    self.value = nn.Linear(dims, hidden)



  def forward(self, data, query):

    key = self.key(data).unsqueeze(0).permute([1,0,2])
    value = self.value(data).unsqueeze(0).permute([1,0,2])

    return self.attention(query.unsqueeze(1), key, value)[0]


class StructNet(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(322, 124),
            nn.LeakyReLU(),
            nn.Linear(124, 86),
            nn.LeakyReLU(),
            nn.Linear(86, 24)
        )
        self.attention = AttentionEncoder(24, 1) 
        self.conv_net = ConvBlock(config)

    def proc_dataset(self, contexts, treatments, effects):
        contexts = contexts.cuda() if self.config.cuda else contexts
        effects = effects.cuda() if self.config.cuda else effects
        treatments = treatments.cuda() if self.config.cuda else treatments

        effects = effects.unsqueeze(1)
        bs = treatments.shape[1]
        treatments = treatments.view(-1, 1)
        effects = torch.repeat_interleave(effects, bs , 1).view(-1, 1)
        context_emb = self.conv_net(contexts, merge=True)

        dataset_tensor = torch.cat((context_emb, treatments, effects), dim=1)
        self.dataset_emb = self.encoder(dataset_tensor)

    def forward(self, contexts):
        Q = self.attention.query(self.conv_net(contexts))
        causal_structs = self.attention(self.dataset_emb, Q)
        return torch.sigmoid(causal_structs)

    


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

        return (no_treat_mu, treat_mu), (no_treat_sigma, treat_sigma )


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

        contexts = contexts.view(-1, *config.dim_in)
        treatments = treatments.flatten()

        mu_pred, sigma_pred = model(contexts, add_delta=True)

        mu_pred = torch.stack(mu_pred).squeeze()
        sigma_pred = torch.stack(sigma_pred).squeeze()

        mu_pred = mu_pred.gather(0, treatments[None]) # 2, num_arms
        sigma_pred = sigma_pred.gather(0, treatments[None])

        if config.fixed_sigma:
            sigma_pred = torch.ones_like(sigma_pred) * 0.1

        sigma_mat_pred = utils.to_diag_var(sigma_pred, cuda=config.cuda)

        loss = -D.MultivariateNormal(mu_pred.squeeze(), sigma_mat_pred).log_prob(effects).mean()

        loss.backward()
        opt.step()

        losses.append(loss.item()) # better way of doing this tho
    
    return losses


def freeze(net : nn.Module, unfreeze=False):
    for param in net.parameters():
        param.requires_grad = unfreeze
