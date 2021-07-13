from dataclasses import dataclass, fields
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
import plotly.express as px
import plotly.graph_objects as go

import torch
from torch.utils import data

from agents.mnist_agent import CmnistBanditAgent
from causal_env.envs import CausalMnistBanditsEnv, causal_mnist
from agents.mnist_agent import CmnistBanditAgent

from collections import defaultdict
from dataclasses import dataclass, field

from typing import Dict, List

@dataclass
class _Telemetry:
    kl: List[float] = field(default_factory=list)
    nll: List[float]= field(default_factory=list)
    regret: List[float]= field(default_factory=list)
    treatment_uncertainty: Dict = field(default_factory=dict)
    no_treatment_uncertainty: Dict = field(default_factory=dict)

    causal_model = List[int]

    def __len__(self):
        return len(self.nll)

class Vis:
    def __init__(self, causal_model:  List[int]) -> None:
        self.store = _Telemetry(causal_model=causal_model)
    
    def collect(self, agent: CmnistBanditAgent, env:CausalMnistBanditsEnv):
        
        # at some point I am going to have to compute this for baseline agents and other agents

        contexts = torch.stack([env.sample_mnist(n) for n in range(env.config.num_arms)])
        uncertainty = agent.calculate_uncertainties(contexts) # cuda???
        uncertainty = uncertainty.cpu().numpy()

        uncertainty.size(1) # should be num arms

        for n in range(env.config.num_arms):
            self.store.no_treatment_uncertainty[n].append(uncertainty[0, n].item())
            self.store.treatment_uncertainty[n].append(uncertainty[1, n].item())

        loss = agent.history.loss[-1]
        kl = env.compute_kl(agent)
        regret = env.compute_regret(agent)

        self.store.kl.append(kl)
        self.store.nll.append(loss)
        self.store.regret.append(regret)

    def plot_loss(self, title='Negative Log-Likelihood'):
        fig = px.line(y = self.store.nll, x = list(range(len(self.store))), title=title)
        return fig

    def plot_uncertainty(self):
        treatment_fig = go.Figure()
        arms = len(self.store.treatment_uncertainty.keys())
        x = list(range(len(self.store)))

        for arm in range(arms):
            ys = self.store.treatment_uncertainty[arm]
            treatment_fig.add_trace(go.Scatter(x=x, y=ys,
                                name=f'Arm #{arm} (causal={arm in self.store.causal_model})',
                                mode='lines+markers',
                                name='lines+markers'))
        no_treatment_fig = go.Figure()
        arms = len(self.store.treatment_uncertainty.keys())
        x = list(range(len(self.store)))

        for arm in range(arms):
            ys = self.store.no_treatment_uncertainty[arm]
            no_treatment_fig.add_trace(go.Scatter(x=x, y=ys,
                                name=f'Arm #{arm} (causal={arm in self.store.causal_model})',
                                mode='lines+markers',
                                name='lines+markers'))
        
        treatment_fig.update_layout(title='Epistemic Uncerstainty for Treatment')
        no_treatment_fig.update_layout(title='Epistemic Uncerstainty for No Treatment')

        return treatment_fig, no_treatment_fig

    def plot_kl(self, title='KL-divergence'):
        fig = px.line(y = self.store.kl, x = list(range(len(self.store))), title=title)
        return fig

    def plot_regret(self, title='Agent Regret'):
        fig = px.line(y = self.store.regret, x = list(range(len(self.store))), title=title)
        return fig




if __name__ == "__main__":
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    fig.show()
    