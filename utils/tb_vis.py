import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy.stats import norm

from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.base_agent import BaseAgent


def safenumpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
class TensorBoardVis:
    def __init__(self, config) -> None:
        self.config = config
        self.writer = SummaryWriter(log_dir=config.telemetry_dir)
        print(config)


    def collect(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        unc = agent.compute_digit_uncertainties(env.digit_contexts, timestep.id)    

        if unc is not None:
            unc = safenumpy(unc.view(env.config.num_arms, 2))
            # transforming tensor to dict
            treat_unc = {f'arm #{i}': unc[i, 0] for i in range(env.config.num_arms)}
            no_treat_unc = {f'arm #{i}': unc[i, 0] for i in range(env.config.num_arms)}

            self.writer.add_scalars('Treatment/uncertainty', tag_scalar_dict=treat_unc, global_step=timestep.id)
            self.writer.add_scalars('NoTreatment/uncertainty', tag_scalar_dict=no_treat_unc, global_step=timestep.id)

        regret = env.compute_regret(agent)
        self.writer.add_scalar('General/Regret', regret , global_step=timestep.id)

    def record_normal(self, tag, mean, variance, t):
        self.writer.add_histogram(tag, self.unroll_norm(mean, variance), global_step=t)


    @staticmethod
    def unroll_norm(mu, sigma):
        xs = np.linspace(-3, 3, num=200)
        return norm.pdf(xs, mu, sigma)


    def collect_arm_distributions(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        means, variances = agent.compute_digit_distributions(env.digit_contexts)
        means = safenumpy(means.view(env.config.num_arms, 2))
        variances = safenumpy(variances.view(env.config.num_arms, 2))

        for i in range(env.config.num_arms):
            mu, sigma = means[i, 1], variances[i, 1]
            self.record_normal(f'Treatment Agent Distributions/arm #{i}', mu, sigma, timestep.id)

            mu, sigma = means[i, 0], variances[i, 0]
            self.record_normal(f'No Treatment Agent Distributions/arm #{i}', mu, sigma, timestep.id)

    def record_distributions(self,env, parent_tag, means, variances):
        means = safenumpy(means.view(env.config.num_arms, 2))
        variances = safenumpy(variances.view(env.config.num_arms, 2))

        for i in range(env.config.num_arms):
            mu, sigma = means[i, 1], variances[i, 1]
            self.record_normal(parent_tag + ' Treatment', mu, sigma, i)

            mu, sigma = means[i, 0], variances[i, 0]
            self.record_normal(parent_tag + ' No Treatment', mu, sigma, i)
