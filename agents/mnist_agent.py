import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class MnistBanditAgent:
    def __init__(self, ):
        # learning is the ITE of causal arms
        self.episode_buffer = []
        self.possible_models = []

        # needs to determine ITE
        # needs p(r | x, t, w) for each of the digits
        

    def choose_intervention(self):
        pass

    def observe(self, actions, reward):
        # update parameters, calculate likelihoods
        # calculate the structural parameter
        pass

    
    