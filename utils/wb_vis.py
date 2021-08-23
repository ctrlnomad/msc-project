from datetime import time
import torch

import numpy as np
from scipy.stats import norm

from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.base_agent import BaseAgent

import wandb

from utils.misc import safenumpy
class WBVis:
    def __init__(self, config, agent, env) -> None:
        self.config = config

        self.run = wandb.init(project='causal-bandits', 
            entity='ctrlnomad', 
            config=config,
            notes=str(env))

        self.run.watch(agent.estimator.models, log='ALL',  log_freq=5)

        def log_dict(tag, tensor, step):
            d = dict()
            for i in range(config.num_arms):
                key = f'causal-arm-#{i}' if i in env.causal_ids else f'arm-#{i}'
                d[f'{tag}/{key}'] = tensor[i]
            self.run.log(d, step=step)

        self.log_dict = log_dict
    

    def collect(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        unc = agent.compute_digit_uncertainties(env.digit_contexts)    
        # beliefs

        # losses 


        if unc is not None:
            unc = safenumpy(unc)
            # transforming tensor to dict

            self.log_dict('Treatment/Uncertainty', unc[1,:], step=timestep.id)
            self.log_dict('NoTreatment/Uncertainty', unc[0,:], step=timestep.id)


        regret = env.compute_regret(agent)
        wandb.log({'General/Regret': regret}, step=timestep.id) # useless for now

        # errors
        ite_pred = agent.estimator(env.digit_contexts)[0]
        env_ite = env.ite.cuda() if self.config.cuda else env.ite 
        errors = env_ite - ite_pred

        self.log_dict('Treatment/MSE Error', errors[1, :], step=timestep.id)
        self.log_dict('NoTreatment/MSE Error', errors[0, :], step=timestep.id)

        
        mse = torch.pow(errors, 2).mean()
        wandb.log({'General/ITE MSE': mse}, step=timestep.id)


    def collect_arm_distributions(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        means, variances = agent.compute_digit_distributions(env.digit_contexts)
        means = safenumpy(means)
        variances = safenumpy(variances)

        self.log_dict('Treatment/Means', means[1, :],step=timestep.id)
        self.log_dict('NoTreatment/Means', means[0, :],step=timestep.id)
        self.log_dict('Treatment/Variances', variances[1, :],step=timestep.id)
        self.log_dict('NoTreatment/Variances', variances[0, :],step=timestep.id)

    def collect_beliefs(self,  env,  agent):
        beliefs = agent.estimator.compute_beliefs(env.digit_contexts)
        self.run.log({'Causal Model Beliefs': wandb.Histogram(beliefs)})

    def finish(self):
        wandb.finish()
