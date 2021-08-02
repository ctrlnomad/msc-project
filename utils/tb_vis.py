import torch
from torch.utils.tensorboard import SummaryWriter


from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.base_agent import BaseAgent


class TensorBoardVis:
    def __init__(self, config) -> None:
        self.config = config
        self.writer = SummaryWriter(log_dir=config.telemetry_dir)
        print(config)


    def collect(self, agent: BaseAgent, env: CausalMnistBanditsEnv, timestep: Timestep):
        unc = agent.compute_digit_uncertainties(env.digit_contexts, timestep.id)    

        if unc is not None:
            unc = unc.view(env.config.num_arms, 2)
            # transforming tensor to dict
            treat_unc = {f'arm #{i}': unc[i, 0] for i in range(env.config.num_arms)}
            no_treat_unc = {f'arm #{i}': unc[i, 0] for i in range(env.config.num_arms)}

            self.writer.add_scalars('Treatment/uncertainty', tag_scalar_dict=treat_unc, global_step=timestep.id)
            self.writer.add_scalars('NoTreatment/uncertainty', tag_scalar_dict=no_treat_unc, global_step=timestep.id)

        
        means, variances = agent.compute_digit_distributions(env.digit_contexts)

        if means is not None and variances is not None:
            means = means.view(env.config.num_arms, 2)
            variances = variances.view(env.config.num_arms, 2)

            treat_means = {f'arm #{i}': means[i, 1] for i in range(env.config.num_arms)}
            treat_vars = {f'arm #{i}': variances[i, 1] for i in range(env.config.num_arms)}

            no_treat_means = {f'arm #{i}': means[i, 0] for i in range(env.config.num_arms)} 
            no_treat_vars = {f'arm #{i}': variances[i, 0] for i in range(env.config.num_arms)}

            self.writer.add_scalars('Treatment/means', tag_scalar_dict=treat_means, global_step=timestep.id)
            self.writer.add_scalars('Treatment/variances', tag_scalar_dict=treat_vars, global_step=timestep.id)


            self.writer.add_scalars('NoTreatment/means', tag_scalar_dict=no_treat_means, global_step=timestep.id)
            self.writer.add_scalars('NoTreatment/variances', tag_scalar_dict=no_treat_vars, global_step=timestep.id)


        regret = env.compute_regret(agent)
        self.writer.add_scalar('General/Regret', regret , global_step=timestep.id)

        # todo add add_histogram(tag, values, global)