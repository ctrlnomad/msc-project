import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class BayesianConvNet(nn.Module):
    """
    should have two brancches for two treatment effects
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.view(-1, 320)
        x = torch.relu(self.fc1(x))
        # two separate confounders? 
        x = torch.dropout(x, training=True) # TODO: does just this make it a bayesian network?
        x = self.fc2(x)

    def compute_entropy(x, n_samples):
        # compute entropy with dropout
        pass

class CmnistBanditAgent:
    """
    ITE network p(r | x, t, w). The observational data is confounded by other treatments
    ITE or CATE? should I ask?
    and variance that is not fixed 
    
    loss = -dist.normal(mu_theta(x), sigma_theta(x)).log_prob(y)

    half the round we learn by empty interventions
    the other times we learn by choosing based on epsitemic uncertainty
    we evaluate by regret. 
    """
    def __init__(self, ):
        # learning is the ITE of causal arms
        self.episode_buffer = []
        self.possible_models = []

        # needs to determine ITE
        # needs p(r | x, t, w) for each of the digits
        
        self.ite_network = BayesianConvNet

    def choose_intervention(self):
        pass

    def observe(self, actions, reward):
        # update parameters, calculate likelihoods
        # calculate the structural parameter
        pass

    
    