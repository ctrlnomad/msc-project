from typing import NamedTuple
import gym
import gym.spaces as spaces
import numpy as np

from dataclasses import dataclass
from numpy.core.fromnumeric import nonzero

import torch 
import torchvision

from collections import namedtuple

@dataclass
class CausalMnistBanditConfig:
    num_arms: int



Timestep = namedtuple('Timestep', ['_context', 'high_dim_context', 'treatments', 'reward', 'id'])

class CausalMnistBanditsEnv(gym.Env):
    """
    action space is multibinary. 
    the observations that the agent recieves are MNIST digits of size (128, num_arms)
    the environment samples these randomly at each time step and the agent's job is to find out the ITE (maybe extend to p(y | x, t, w))
    the effect on the reward is associated with the digit, not the index of the arm 

    The agent can intervene by selecting to pull an arm or choose to do nothing. 
    The other arms are set to 0,1 depending on a biased coin flip. 
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config


        self.action_space = spaces.Discrete(self.config.num_arms + 1)

        self.observation_space = spaces.Box(0, 122, (self.config.num_arms, 28, 28) ) # what about the other arms observation

        self.default_probs = np.random.rand(self.config.num_arms)

        self.digit_ITE = np.random.rand(self.config.num_arms) *2-1

        self.mnist_dataset = torchvision.datasets.MNIST('/data/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))

    def sample_mnist(self, num):
        # samples an image of num from the MNIST dataset
        idxs = (self.mnist_dataset.targets == num).nonzero(as_tuple=True)[0]
        idx = np.random.choice(idxs)
        return self.mnist_dataset[idx][0] # return just the image tensor

    def reset(self) -> Any:
        self.current_timestep = self._make_timestep(1)

    def _make_timestep(self, tsid) -> Any:
        ts = Timestep()
        ts._context = np.random.choice(np.arange(self.config.num_arms), size=self.config.num_arms)
        ts.high_dim_context = torch.stack([self.sample_mnist(n) for n in ts._context])
        ts.treatments = (self.default_probs > np.random.rand(self.config.num_arms)) * 1
        ts.id = tsid
        return ts


    def seed(self, seed: int) -> List[int]:
        return super().seed(seed=seed)

    def step(self, action: _ActionType) -> Tuple[_OperationType, float, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        if action != self.config.num_arms + 1:
            self.current_timestep.treatments[action] = 1
        
        reward_mean = self.digit_ITE[self.current_timestep.treatments].sum()

        return np.random.normal(reward_mean, 1) # constant variance





class MetaCausalMnistBanditsEnv(gym.Env):
    pass