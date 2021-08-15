from datetime import time
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy.stats import norm

from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.base_agent import BaseAgent

from utils.misc import safenumpy

class TensorBoardVis:
    def __init__(self, config) -> None:
        self.config = config
        self.writer = SummaryWriter(log_dir=config.telemetry_dir)

    def record_experiment(self,  env: CausalMnistBanditsEnv, agent: BaseAgent, config):

        self.writer.add_text('Env Description', str(env), global_step=0)
        self.writer.add_text('Config', str(config) , global_step=0)

    def make_scalar_dict(self, tensor, causal_ids, num_arms):
        return {f'causal arm #{i}' if i in causal_ids else f'arm #{i}' : tensor[i] for i in range(num_arms)}


    def collect(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        unc = agent.compute_digit_uncertainties(env.digit_contexts)    

        if unc is not None:
            unc = safenumpy(unc)
            # transforming tensor to dict
            treat_unc = self.make_scalar_dict(unc[1, :], env.causal_ids, env.config.num_arms)
            no_treat_unc = self.make_scalar_dict(unc[0, :], env.causal_ids, env.config.num_arms)

            self.writer.add_scalars('Treatment/uncertainty', tag_scalar_dict=treat_unc, global_step=timestep.id)
            self.writer.add_scalars('NoTreatment/uncertainty', tag_scalar_dict=no_treat_unc, global_step=timestep.id)

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

    def collect_cate(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep ):
        
        ite_pred, _ = agent.estimator(env.digit_contexts)
        env_ite = env.ite.cuda() if self.config.cuda else env.ite 
        
        true_cate  =  env_ite[1, :] -   env_ite[0, :]  
        pred_cate  = ite_pred[1, :] -  ite_pred[0, :]

        # cate errors 
        cate_errors = true_cate - pred_cate
        d = self.make_scalar_dict(cate_errors, env.causal_ids, env.config.num_arms)
        self.writer.add_scalars('General/CATE Errors', d, timestep.id)


        cate_mse = torch.pow(cate_errors, 2).mean()
        self.writer.add_scalar('General/CATE MSE', cate_mse, timestep.id)

        # cate uncertainteis 
        cate_unc = agent.estimator.compute_cate_uncertainty(env.digit_contexts)
        d = self.make_scalar_dict(cate_unc, env.causal_ids, env.config.num_arms)
        self.writer.add_scalars('General/CATE Uncertainty', d, timestep.id)




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
