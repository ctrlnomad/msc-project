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
import utils.mnist as mnist

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
        self.seed(config.seed)

        self.action_space = spaces.Discrete(self.config.num_arms * 2 + 1)
        self.noop = self.config.num_arms*2

        self.observation_space = spaces.Box(0, 122, (self.config.num_arms, 28, 28) ) # what about the other arms observation

        self.default_probs = torch.rand(self.config.num_arms)
        self.default_dist = distributions.Bernoulli(probs=self.default_probs)
    
        self.causal_ids = np.random.choice(np.arange(config.num_arms), size=config.causal_arms, replace=False)
        
        self.causal_model = torch.zeros(self.config.num_arms)
        self.causal_model[self.causal_ids] = 1

        self.ite = torch.zeros((2, config.num_arms))
        self.ite[:, self.causal_ids] = torch.rand((2, self.config.causal_arms))*2-1
        self.variance = torch.rand((2, self.config.num_arms))*2

        self.digit_sampler = mnist.MnistSampler()

        logger.info('environment inited')
        self._inited = True

    @property
    def digit_contexts(self):
        if self.config.cuda:
            return self.digit_sampler.sample_array(self.config.num_arms).cuda()
        else:
            return self.digit_sampler.sample_array(self.config.num_arms)


    def reset(self) -> Any:
        self.current_timestep = self._make_timestep(1)
        return self.current_timestep

    def _make_timestep(self, tsid) -> Any:
        ts = Timestep()
        ts.info = np.random.choice(np.arange(self.config.num_arms), size=self.config.num_arms)
        ts.context = torch.stack([self.digit_sampler.sample(n) for n in ts.info])
        ts.info = torch.LongTensor(ts.info)
        ts.treatments = self.default_dist.sample().long()
        ts.done = tsid >= self.config.num_ts
        ts.id = tsid
        return ts


    def seed(self, seed: int):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def step(self, action) -> Timestep:
        assert self.action_space.contains(action)
        logger.info(f'recieved action with id [{action}] [noop={action ==self.noop}]')

        if action != self.noop: 
            arm_id = action - self.config.num_arms if action >= self.config.num_arms else action
            intervention = arm_id > self.config.num_arms
            self.current_timestep.treatments[arm_id] = int(intervention)

        
        current_ites = self.ite.index_select(1, self.current_timestep.info)
        current_variances = self.variance.index_select(1, self.current_timestep.info)

        reward_mean = current_ites.gather(0, self.current_timestep.treatments[None]).squeeze()
        reward_variances = current_variances.gather(0, self.current_timestep.treatments[None]).ravel()

        diag_vars = utils.to_diag_var(reward_variances)

        reward = distributions.MultivariateNormal(reward_mean, diag_vars).sample().sum()

        # generate new tiemstep
        old_timestep = self.current_timestep
        old_timestep.reward = reward
        self.current_timestep = self._make_timestep(self.current_timestep.id + 1)

        logger.info(f'[{old_timestep.id}] timestep: \n\t{old_timestep}')
        return old_timestep, self.current_timestep 

    def compute_kl(self, agent):
        # sample each digit
        mu_pred, sigma_pred = agent.effect_estimator(self.digit_contexts)

        mu_pred = mu_pred.ravel()
        sigma_pred = utils.to_diag_var(sigma_pred.ravel())

        mu_true = self.ite.ravel()
        sigma_true = utils.to_diag_var(self.variance.ravel())
        
        pred_dist = distributions.MultivariateNormal(mu_pred, sigma_pred)
        true_dist = distributions.MultivariateNormal(mu_true, sigma_true)

        kl = distributions.kl_divergence(true_dist, pred_dist)
        return kl.mean().item()

    def compute_regret(self, agent):
        best_action = agent.compute_best_action(self.digit_contexts)
        best_action = utils.safenumpy(best_action)

        arm = best_action[1]
        treatment = best_action[0]

        regret = (self.ite.max() - self.ite[treatment, arm])
        return regret.item()

