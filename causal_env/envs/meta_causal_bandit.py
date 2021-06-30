import gym
import gym.spaces as spaces
import numpy as np

from dataclasses import dataclass


@dataclass
class MetaCausalBanditConfig:
    episode_len: int

    total_arms: int
    causal_arms: int

    num_tasks: int


class MetaCausalBanditsEnv(gym.Env):
    def __init__(self, config: MetaCausalBanditConfig):
        self.config = config
        self.action_space = spaces.Discrete(config.total_arms)
        self.observation_space = spaces.Box(0,1, 1)

        self.tasks = []
        for _ in range(config.num_tasks):
            model = self._make_causal_task()
            self.tasks.append(model)
        
        self.reset()


    def _make_causal_task(self):
        causal_arms = np.random.choice(np.arange(self.config.total_arms), size=self.config.causal_arms, replace=False)
        causal_model = np.zeros(self.config.total_arms)
        causal_model[causal_arms] = 1
        treatment_effect = np.tanh(np.random.rand(self.config.total_arms)*2 - 1) # TODO, investigate
        causal_model *= treatment_effect 
        return causal_model

    @property
    def causal_models(self): # for the agent to access
        return [np.nonzero(t) for t in self.tasks]

    def reset(self):
        self.current_task = np.random.choice(self.tasks)
        self.timestep = 0

    def seed(self, seed: int) -> List[int]:
        return super().seed(seed=seed)

    def step(self, arm):
        # bit of a random thing right
        # usually the reward is distributed with a mean of 0
        # the causal arms cause the mean to increase
        # check about the number of trials 
        self.timestep += 1
        return self.timestep, np.random.normal(loc=self.current_task[arm]), self.timestep >= self.config.episode_len, dict(self.config)