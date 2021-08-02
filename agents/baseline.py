
import numpy as np
from agents.base_agent import BaseAgent

# UCB and Thompson adapted fom https://github.com/WhatIThinkAbout

def random_argmax(value_list):
  """ a random tie-breaking argmax"""
  values = np.asarray(value_list)
  return np.argmax(np.random.random(values.shape) * (values==values.max()))



class PowerSocket:
    """ the base power socket class """
    
    def __init__(self, q):                
        self.q = q        # the true reward value              
        self.initialize() # reset the socket
        
    def initialize(self):
        self.Q = 0   # the estimate of this socket's reward value                
        self.n = 0   # the number of times this socket has been tried        
    
    def charge(self):
        """ return a random amount of charge """
        
        # the reward is a guassian distribution with unit variance around the true
        # value 'q'
        value = np.random.randn() + self.q        
        
        # never allow a charge less than 0 to be returned        
        return 0 if value < 0 else value
                    
    def update(self,R):
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


class UCBAgent(BaseAgent):
    def __init__(self, num_actions: int) -> None:
        
        self.sockets = {i: UCBSocket() for i in range(num_actions)}

    def act(self, timestep):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    @property
    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        raise NotImplementedError()

    @property
    def compute_digit_distributions(self, contexts: torch.Tensor):
        raise NotImplementedError()
            
    @property
    def compute_best_action(self, contexts: torch.Tensor):
        raise NotImplementedError()
class GaussianThompsonSocket(PowerSocket):
    def __init__(self, q):                
                
        self.τ_0 = 0.0001  # the posterior precision
        self.μ_0 = 1       # the posterior mean
        
        # pass the true reward value to the base PowerSocket             
        super().__init__(q)         
        
    def sample(self):
        """ return a value from the the posterior normal distribution """
        return (np.random.randn() / np.sqrt(self.τ_0)) + self.μ_0    
                    
    def update(self,R):
        """ update this socket after it has returned reward value 'R' """   

        # do a standard update of the estimated mean
        super().update(R)    
               
        # update the mean and precision of the posterior
        self.μ_0 = ((self.τ_0 * self.μ_0) + (self.n * self.Q))/(self.τ_0 + self.n)        
        self.τ_0 += 1 
    
class ThompsonAgent(BaseAgent):
    
    def act(self, timestep):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    @property
    def compute_digit_uncertainties(self, contexts: torch.Tensor):
        raise NotImplementedError()

    @property
    def compute_digit_distributions(self, contexts: torch.Tensor):
        raise NotImplementedError()
            
    @property
    def compute_best_action(self, contexts: torch.Tensor):
        raise NotImplementedError()