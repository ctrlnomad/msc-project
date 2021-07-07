import gym
import gym.spaces as spaces
import numpy as np

from dataclasses import dataclass

import torch 
import torchvision
import torch.distributions as distributions
import torch.utils.data as trchdata
from typing import Any, Sequence, Tuple, Dict, List


import logging
logger = logging.getLogger(__name__)

@dataclass
class CausalMnistBanditsConfig:
    num_arms: int
    causal_arms: int

    num_ts: int

@dataclass
class Timestep:
    info: Any = None
    context: torch.Tensor = None
    treatments: torch.Tensor = None
    reward: float = None
    done: bool = False
    id: int = None

class TimestepDataset(trchdata.Dataset):

    def __init__(self, dataset: Timestep) -> None:
        super().__init__()
        self.dataset = dataset
    
    @classmethod
    def build(memory: List[Timestep]):
        dataset = Timestep(*zip(memory))
        return TimestepDataset(dataset)

    def __get_item___(self, i):
        return self.dataset.context[i], \
            self.dataset.treatments[i], \
            self.dataset.reward[i]

    def __len__(self):
        return len(self.dataset.rewards)

class CausalMnistBanditsEnv(gym.Env):
    """
    action space is multibinary. 
    the observations that the agent recieves are MNIST digits of size (128, num_arms)
    the environment samples these randomly at each time step and the agent's job is to find out the ITE (maybe extend to p(y | x, t, w))
    the effect on the reward is associated with the digit, not the index of the arm 

    The agent can intervene by selecting to pull an arm or choose to do nothing. 
    The other arms are set to 0,1 depending on a biased coin flip. 
    TODO:
        - ITE is x_1 * a - x_2
        - option for non-stationary variance 
    """
    def init(self, config: CausalMnistBanditsConfig) -> None:
        super().__init__()

        self.config = config

        self.action_space = spaces.Discrete(self.config.num_arms + 1)

        self.observation_space = spaces.Box(0, 122, (self.config.num_arms, 28, 28) ) # what about the other arms observation

        self.default_probs = np.random.rand(self.config.num_arms)
    
        causal_ids = np.random.choice(np.arange(config.num_arms), size=config.causal_arms, replace=False)


        self.ite = np.zeros(2, config.num_arms) 
        self.ite[:, causal_ids] = np.random.rand(2, self.config.causal_arms)*2-1
        self.variance = np.random.rand(2, config.num_arms) * 2

        self.mnist_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
        self.seed(config.seed)
        logger.info('environment inited')
        self._inited = True

    def sample_mnist(self, num):
        # samples an image of num from the MNIST dataset
        idxs = (self.mnist_dataset.targets == num).nonzero(as_tuple=True)[0]
        idx = np.random.choice(idxs)
        return self.mnist_dataset[idx][0] # return just the image tensor

    def reset(self) -> Any:
        self.current_timestep = self._make_timestep(1)
        return self.current_timestep

    def _make_timestep(self, tsid) -> Any:
        ts = Timestep()
        ts.info = np.random.choice(np.arange(self.config.num_arms), size=self.config.num_arms)
        ts.context = torch.stack([self.sample_mnist(n) for n in ts.context]).squeeze()
        ts.treatments = (self.default_probs > np.random.rand(self.config.num_arms)) * 1
        ts.done = tsid >= self.config.num_ts
        ts.id = tsid
        return ts


    def seed(self, seed: int) -> List[int]:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def step(self, action) -> Timestep:
        assert self.action_space.contains(action)

        if action != self.config.num_arms:
            self.current_timestep.treatments[action] = 1
        
        reward_mean = self.ite[self.current_timestep.treatments, :]
        reward_variances = self.variances[self.current_timestep.treatments, :]

        diag_vars = torch.zeros(len(self.current_timestep.treatments))
        diag_vars[torch.eye(len(self.current_timestep.treatments))] = reward_variances

        reward = distributions.MultivariateNormal(reward_mean, reward_variances).sample()

        # generate new tiemstep
        self.current_timestep = self._make_timestep(self.current_timestep.id + 1)
        self.current_timestep.reward = reward

        return self.current_timestep 





class MetaCausalMnistBanditsEnv(gym.Env):
    pass