from genericpath import exists
from agents.proba_agent import VariationalAgentConfig
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from numpy.core.arrayprint import TimedeltaFormat
import seaborn as sns
import pandas as pd
from torch.utils import data

sns.set_style()

import torch

from causal_env.envs import CausalMnistBanditsEnv
from causal_env.envs import Timestep
from agents.base_agent import BaseAgent

from typing import Dict, List
from collections import defaultdict, namedtuple

from pathlib import Path
from utils.mnist import MnistSampler

@dataclass
class _Metric:
    x: List[int] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    def plot(self, title=''):
        plt.figure()
        ax = sns.lineplot(y = self.y, x = self.x)
        ax.set_title(title)
        return ax.figure

@dataclass
class _GeneralTelemetry:
    kl: _Metric = field(default_factory=_Metric)
    nll: _Metric = field(default_factory=_Metric)
    regret: _Metric = field(default_factory=_Metric)

    rewards: _Metric = field(default_factory=_Metric)
    action_frequency: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))


@dataclass
class _SpecificTelemetry: # each telemetry object is like a directory inside of the figure folder
    causal_model : List[int]

    uncertainty: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))
    variance: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))
    means: Dict[int, _Metric] = field(default_factory=lambda: defaultdict(_Metric))

    def plot_dict_metric(self, data:  Dict[int, _Metric], title=''):
        plt.figure()
        for k, v in data.items():
            args = {
                 'label':f'arm #{k}'
            }
            if k in self.causal_model:
                args['linestyle'] = 'dashed'
            plt.plot(v.x, v.y, **args)
        
        plt.legend()
        plt.title(title)
        fig = plt.gcf() 
        return fig


class Vis:
    def __init__(self, path: str, causal_model:  List[int]) -> None: # different telemetries?? like general and specific
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

        self.store = _GeneralTelemetry()

        self.treatment_store = _SpecificTelemetry(causal_model)
        self.no_treatment_store = _SpecificTelemetry(causal_model)

        self.sampler = MnistSampler()
    
    def collect(self, agent: BaseAgent, env:CausalMnistBanditsEnv, timestep: Timestep):
        means, variances = agent.compute_digit_distributions(env.digit_contexts)
        uncertainty = agent.compute_digit_uncertainties(env.digit_contexts)

        for n in range(env.config.num_arms):
            self.no_treatment_store.uncertainty[n].y.append(uncertainty[n, 0])
            self.treatment_store.uncertainty[n].y.append(uncertainty[n, 1])

            self.no_treatment_store.uncertainty[n].x.append(timestep.id)
            self.treatment_store.uncertainty[n].x.append(timestep.id)

            self.no_treatment_store.variance[n].y.append(variances[n, 0])
            self.treatment_store.variance[n].y.append(variances[n, 1])

            self.no_treatment_store.variance[n].x.append(timestep.id)
            self.treatment_store.variance[n].x.append(timestep.id)

            self.no_treatment_store.means[n].y.append(variances[n, 0])
            self.treatment_store.means[n].y.append(variances[n, 1])

            self.no_treatment_store.means[n].x.append(timestep.id)
            self.treatment_store.means[n].x.append(timestep.id)

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

    def save_plots(self):
        self.store.nll.plot('Negative Log-Likelihood').savefig(self.path / "loss_plot.png")
        self.store.kl.plot('KL-divergence').savefig(self.path / "kl_plot.png")
        self.store.regret.plot('Agent Regret').savefig(self.path / "regret_plot.png")

        no_treatment_path = self.path / 'no_treatment' 
        no_treatment_path.mkdir(exist_ok=True)

        f = self.no_treatment_store.plot_dict_metric(self.no_treatment_store.uncertainty)
        f.savefig(no_treatment_path / 'uncertainty_plot.png')

        f = self.no_treatment_store.plot_dict_metric(self.no_treatment_store.means)
        f.savefig(no_treatment_path / 'means_plot.png')

        f = self.no_treatment_store.plot_dict_metric(self.no_treatment_store.variace)
        f.savefig(no_treatment_path / 'variace_plot.png')

        treatment_path = self.path / 'treatment' 
        treatment_path.mkdir(exist_ok=True)

        f = self.treatment_store.plot_dict_metric(self.treatment_store.uncertainty)
        f.savefig(treatment_path / 'uncertainty_plot.png')

        f = self.treatment_store.plot_dict_metric(self.treatment_store.means)
        f.savefig(treatment_path / 'means_plot.png')

        f = self.treatment_store.plot_dict_metric(self.treatment_store.variace)
        f.savefig(treatment_path / 'variace_plot.png')
