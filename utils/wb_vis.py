from datetime import time
import torch

import numpy as np
from scipy.stats import norm

from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.base_agent import BaseAgent

import wandb

from utils.misc import safenumpy

class TensorBoardVis:
    def __init__(self, config) -> None:
        self.config = config
        wandb.config = config

        wandb.init(project='causal-bandits', entity='ctrlnomad')


    def record_experiment(self,  env: CausalMnistBanditsEnv, agent: BaseAgent, config):

        wandb.log({'Env Description': str(env)})
        wandb.log({'Config': str(config) })


    def make_scalar_dict(self, tensor, causal_ids, num_arms):
        return {f'causal arm #{i}' if i in causal_ids else f'arm #{i}' : tensor[i] for i in range(num_arms)}

    
    def collect(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        unc = agent.compute_digit_uncertainties(env.digit_contexts)    

        if unc is not None:
            unc = safenumpy(unc)
            # transforming tensor to dict
            treat_unc = self.make_scalar_dict(unc[1, :], env.causal_ids, env.config.num_arms)
            no_treat_unc = self.make_scalar_dict(unc[0, :], env.causal_ids, env.config.num_arms)

            wandb.log('Treatment/uncertainty', tag_scalar_dict=treat_unc, global_step=timestep.id)
            wandb.log('NoTreatment/uncertainty', tag_scalar_dict=no_treat_unc, global_step=timestep.id)

        regret = env.compute_regret(agent)
        self.writer.add_scalar('General/Regret', regret , global_step=timestep.id) # useless for now

        # errors
        ite_pred, _ = agent.estimator(env.digit_contexts)
        env_ite = env.ite.cuda() if self.config.cuda else env.ite 
        errors = env_ite - ite_pred

        treat_err = self.make_scalar_dict(errors[1, :], env.causal_ids, env.config.num_arms)
        no_treat_err = self.make_scalar_dict(errors[0, :], env.causal_ids, env.config.num_arms)

        self.writer.add_scalars('Treatment/ITE Errors', tag_scalar_dict=treat_err, global_step=timestep.id)
        self.writer.add_scalars('NoTreatment/ITE Errors', tag_scalar_dict=no_treat_err, global_step=timestep.id)
        
        mse = torch.pow(errors, 2).mean()
        self.writer.add_scalar('General/ITE MSE', mse, timestep.id)


    def collect_arm_distributions(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        means, variances = agent.compute_digit_distributions(env.digit_contexts)
        means = safenumpy(means)
        variances = safenumpy(variances)

        treat_mu = self.make_scalar_dict(means[1, :], env.causal_ids, env.config.num_arms)
        no_treat_mu = self.make_scalar_dict(means[0, :], env.causal_ids, env.config.num_arms)

        treat_sigma =  self.make_scalar_dict(variances[1, :], env.causal_ids, env.config.num_arms)
        no_treat_sigma =  self.make_scalar_dict(variances[0, :], env.causal_ids, env.config.num_arms)


        self.writer.add_scalars(f'Treatment/Means',treat_mu, timestep.id)
        self.writer.add_scalars(f'NoTreatment/Means',no_treat_mu, timestep.id)

        self.writer.add_scalars(f'Treatment/Variances', treat_sigma, timestep.id)
        self.writer.add_scalars(f'NoTreatment/Variances', no_treat_sigma, timestep.id)

    def record_distributions(self,env, parent_tag, means, variances):
        means = safenumpy(means.view(env.config.num_arms, 2))
        variances = safenumpy(variances.view(env.config.num_arms, 2))

        for i in range(env.config.num_arms):
            mu, sigma = means[i, 1], variances[i, 1]
            self.record_normal(parent_tag + ' Treatment', mu, sigma, i)

            mu, sigma = means[i, 0], variances[i, 0]
            self.record_normal(parent_tag + ' No Treatment', mu, sigma, i)
