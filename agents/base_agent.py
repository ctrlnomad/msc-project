class BaseAgent:
    def __init__(self) -> None:
        pass
    
    @property
    def best_action(self):
        raise NotImplementedError()

    def act(self, timestep):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

        