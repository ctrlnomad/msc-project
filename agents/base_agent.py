import torch

class BaseAgent:
    def __init__(self) -> None:
        pass

    def act(self, timestep):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        raise NotImplementedError()

    def compute_digit_distributions(self, contexts: torch.Tensor):
        raise NotImplementedError()
            
    def compute_best_action(self, contexts: torch.Tensor):
        raise NotImplementedError()