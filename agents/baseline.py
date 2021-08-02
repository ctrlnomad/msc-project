
import numpy as np
import torch
from agents.base_agent import BaseAgent
from causal_env.envs.timestep import Timestep


# UCB and Thompson adapted fom https://github.com/WhatIThinkAbout
def random_argmax(value_list):
  """ a random tie-breaking argmax"""
  values = np.asarray(value_list)
  return np.argmax(np.random.random(values.shape) * (values==values.max()))



class PowerSocket:
    """ the base power socket class """
    
    def __init__(self):                            
        self.initialize() # reset the socket
        
    def initialize(self):
        self.Q = 0   # the estimate of this socket's reward value                
        self.n = 0   # the number of times this socket has been tried        
                    
    def update(self, R):
        """ update this socket after it has returned reward value 'R' """     
    
        # increment the number of times this socket has been tried
        self.n += 1

        # the new estimate of the mean is calculated from the old estimate
        self.Q = (1 - 1.0/self.n) * self.Q + (1.0/self.n) * R
    
    def sample(self,t):
        """ return an estimate of the socket's reward value """
        return self.Q
            
class UCBSocket(PowerSocket):

    def __init__( self, q, **kwargs ):    
        """ initialize the UCB socket """                  
        
        # store the confidence level controlling exploration
        self.confidence_level = kwargs.pop('confidence_level', 2.0)       
                
        # pass the true reward value to the base PowerSocket   
        super().__init__(q)           
        
    def uncertainty(self, t): 
        """ calculate the uncertainty in the estimate of this socket's mean """
        if self.n == 0: return float('inf')                         
        return self.confidence_level * (np.sqrt(np.log(t) / self.n))         
        
    def sample(self,t):
        """ the UCB reward is the estimate of the mean reward plus its uncertainty """
        return self.Q + self.uncertainty(t)

class GaussianThompsonSocket(PowerSocket):
    def __init__(self):                
                
        self.sigma = 0.0001  # the posterior precision
        self.mu = 1       # the posterior mean
        
        # pass the true reward value to the base PowerSocket             
        
    def sample(self):
        """ return a value from the the posterior normal distribution """
        return (np.random.randn() / np.sqrt(self.sigma)) + self.mu    
                    
    def update(self, R):
        """ update this socket after it has returned reward value 'R' """   

        # do a standard update of the estimated mean
        super().update(R)    
               
        # update the mean and precision of the posterior
        self.mu = ((self.sigma * self.mu) + (self.n * self.Q))/(self.sigma + self.n)        
        self.sigma += 1 
    

class BaselineAgent(BaseAgent):
    def __init__(self, num_actions: int, socket_cls: Union[GaussianThompsonSocket, UCBSocket]) -> None:
        self.sockets = {i: socket_cls() for i in range(num_actions)}

    def act(self, timestep: Timestep):
        qs = [s.sample(timestep.id) for s in self.sockets]
        self.current_action = random_argmax(qs)
        return self.current_action

    def observe(self, timestep: Timestep):
        self.sockets[self.current_action].update(timestep.reward)

    def train(self):
        pass

    def compute_digit_uncertainties(self, contexts: torch.Tensor, t=None):
        if t and  hasattr(self.sockets[0], 'uncertainty'):
            uncertainties = [s.uncertainty(t) for s in self.sockets]
            return uncertainties 
        else:
            return None

    def compute_digit_distributions(self, contexts: torch.Tensor):
        if hasattr(self.sockets[0], 'mu') and hasattr(self.sockets[0], 'sigma'):
            return [s.mu for s in self.sockets],  [s.sigma for s in self.sockets], 
        else:
            return None
            
    def compute_best_action(self, contexts: torch.Tensor):
        return random_argmax([s.Q for  s in range(self.sockets)])