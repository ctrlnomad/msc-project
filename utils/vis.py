from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from numpy.core.arrayprint import TimedeltaFormat
import seaborn as sns
import pandas as pd

sns.set_style()

import torch

from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.proba_agent import VariationalAgent

from typing import Dict, List
from collections import defaultdict, namedtuple

from pathlib import Path
from utils.mnist import MnistSampler

@dataclass
class _Metric:
    x: List[int] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
@dataclass
class _Telemetry:
    causal_model : List[int]
    
    kl: _Metric = field(default_factory=_Metric)
    nll: _Metric = field(default_factory=_Metric)
    regret: _Metric = field(default_factory=_Metric)

    treatment_uncertainty: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))
    no_treatment_uncertainty: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))

    variances: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))
    means: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))

    rewards: _Metric = field(default_factory=_Metric)
    action_frequency: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))

class Vis:
    def __init__(self, causal_model:  List[int]) -> None:
        self.store = _Telemetry(causal_model=causal_model)
        self.sampler = MnistSampler()
    
    def collect(self, agent: VariationalAgent, env:CausalMnistBanditsEnv, timestep: Timestep):
        
        # at some point I am going to have to compute this for baseline agents and other agents

        contexts = torch.stack([self.sampler.sample(n) for n in range(env.config.num_arms)])
        uncertainty = agent.calculate_uncertainties(contexts) # cuda???
        uncertainty = uncertainty.detach().cpu().numpy()

        for n in range(env.config.num_arms):
            self.store.no_treatment_uncertainty[n].y.append(uncertainty[n, 0].item())
            self.store.treatment_uncertainty[n].y.append(uncertainty[n, 1].item())

            self.store.treatment_uncertainty[n].x.append(timestep.id)
            self.store.no_treatment_uncertainty[n].x.append(timestep.id)

        if agent.history.loss:
            loss = agent.history.loss[-1] 
            self.store.nll.y.append(loss)
            self.store.nll.x.append(timestep.id)

        kl = env.compute_kl(agent)
        self.store.kl.y.append(kl)
        self.store.kl.x.append(timestep.id)

        regret = env.compute_regret(agent)
        self.store.regret.y.append(regret)
        self.store.regret.x.append(timestep.id)

    def plot_uncertainty(self):
        plt.figure()
        for k, v in self.store.treatment_uncertainty.items():
            args = {
                 'label':f'arm #{k}'
            }
            if k in self.store.causal_model:
                args['linestyle'] = 'dashed'
            plt.plot(v.x, v.y, **args)
        
        plt.legend()
        plt.title('Treatment Uncertainty')
        treatment_fig = plt.gcf() 
        plt.figure()
        for k, v in self.store.no_treatment_uncertainty.items():
            args = {
                'label':f'arm #{k}'
            }
            if k in self.store.causal_model:
                args['linestyle'] = 'dashed'
            plt.plot(v.x,v.y, **args)
        plt.legend()
        plt.title('No Treatment Uncertainty')
        no_treatment_fig = plt.gcf() 
        return treatment_fig, no_treatment_fig

    def plot_loss(self, title='Negative Log-Likelihood'):
        plt.figure()
        ax = sns.lineplot(y = self.store.nll.y, x = self.store.nll.x)
        ax.set_title(title)
        return ax

    def plot_kl(self, title='KL-divergence'):
        plt.figure()
        ax = sns.lineplot(y = self.store.kl.y, x = self.store.kl.x)
        ax.set_title(title)
        return ax

    def plot_regret(self, title='Agent Regret'):
        plt.figure()
        ax = sns.lineplot(y = self.store.regret.y, x = self.store.regret.x)
        ax.set_title(title)
        return ax

    def save_plots(self, path):
        p = Path(path)
        p.mkdir(exist_ok=True)
        
        self.plot_loss().figure.savefig(p / "loss_plot.png")
        self.plot_kl().figure.savefig(p / "kl_plot.png")
        self.plot_regret().figure.savefig(p / "regret_plot.png")

        treatment_fig, no_treatment_fig = self.plot_uncertainty()
        treatment_fig.savefig(p / '1_uncertainty_plot.png')
        no_treatment_fig.savefig(p / '0_uncertainty_plot.png')