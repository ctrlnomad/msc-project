import gym
import gym.spaces as spaces
import numpy as np

from dataclasses import dataclass, field

import torch 
import torchvision
import torch.distributions as distributions
from typing import Any, List

from causal_env.envs.timestep import Timestep
import utils

import logging
logger = logging.getLogger(__name__)

@dataclass
class CausalMnistBanditsConfig:
    num_arms: int = 10
    causal_arms: int = 4

    num_ts: int = 100_000

class CausalMnistBanditsEnv(gym.Env):
    """
    action space is multibinary. 
    the observations that the agent recieves are MNIST digits of size (128, num_arms)
    the environment samples these randomly at each time step and the agent's job is to find out the ITE (maybe extend to p(y | x, t, w))
    the effect on the reward is associated with the digit, not the index of the arm 

    The agent can intervene by selecting to pull an arm or choose to do nothing. 
    The other arms are set to 0,1 depending on a biased coin flip. 
    """

    def init(self, config: CausalMnistBanditsConfig) -> None:
        super().__init__()

        self.config = config

        self.action_space = spaces.Discrete(self.config.num_arms * 2 + 1) # REMAKE
        self.noop = self.config.num_arms*2

        self.observation_space = spaces.Box(0, 122, (self.config.num_arms, 28, 28) ) # what about the other arms observation

        self.default_probs = torch.rand(self.config.num_arms)
        self.default_dist = distributions.Bernoulli(probs=self.default_probs)
    
        self.causal_ids = np.random.choice(np.arange(config.num_arms), size=config.causal_arms, replace=False)


        self.ite = torch.zeros((2, config.num_arms))
        self.ite[:, self.causal_ids] = torch.rand((2, self.config.causal_arms))*2-1
        self.variance = torch.rand((2, self.config.num_arms))*2

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
        ts.context = torch.stack([self.sample_mnist(n) for n in ts.info])
        ts.treatments = self.default_dist.sample().long()
        ts.done = tsid >= self.config.num_ts
        ts.id = tsid
        return ts


    def seed(self, seed: int) -> List[int]:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def step(self, action) -> Timestep:
        assert self.action_space.contains(action)

        if action != self.noop:
            arm_id = self.config.num_arms - action - 1
            intervention = arm_id > 0
            self.current_timestep.treatments[abs(arm_id)] = int(intervention)

        reward_mean = self.ite.gather(0, self.current_timestep.treatments[None]).squeeze()
        reward_variances = self.variance.gather(0, self.current_timestep.treatments[None]).ravel()

        diag_vars = utils.to_diag_var(reward_variances)

        reward = distributions.MultivariateNormal(reward_mean, diag_vars).sample().sum() # not the only way to generate reward

        # generate new tiemstep
        old_timestep = self.current_timestep
        old_timestep.reward = reward
        self.current_timestep = self._make_timestep(self.current_timestep.id + 1)

        logger.info(f'[{old_timestep.id}] timestep: \n\t{old_timestep}')
        return old_timestep, self.current_timestep 

    def compute_kl(self, agent):
        # sample each digit
        contexts = torch.stack([self.sample_mnist(n) for n in range(self.config.num_arms)])

        mu_pred, sigma_pred = agent.effect_estimator(contexts)

        mu_pred = mu_pred.ravel()
        sigma_pred = utils.to_diag_var(sigma_pred.ravel()) # TODO: check with supervisors

        mu_true = self.ite.ravel()
        sigma_true = utils.to_diag_var(self.variance.ravel())
        
        pred_dist = distributions.MultivariateNormal(mu_pred, sigma_pred)
        true_dist = distributions.MultivariateNormal(mu_true, sigma_true)

        kl = distributions.kl_divergence(true_dist, pred_dist)
        return kl.mean().item() # possibly 

    def compute_regret(self, agent):
        # arm, treatment
        # return self.ite.max() - self.ite[treatment, arm]
        return 0
        

class MetaCausalMnistBanditsEnv(gym.Env):
    pass