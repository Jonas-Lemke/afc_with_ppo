"""
This file contains the actor and critic network for the PPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

# import torch.optim as optim
# from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    """ Policy Model"""
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorNetwork, self).__init__()

        # Define Actor Network (maybe use Tanh() instead of ReLU() for the hidenlayer)
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
    
    def forward(self, state):
        """Runs a forward pass on the NN"""
        logit = self.actor(state)
        
        # # add noise for exploration
        # logit += 0.5 * torch.randn_like(logit)
        
        # # clip to prevent fully determenistic logits (e.g. always 0 or 1)
        # logit = torch.clamp(logit, -10, 10)

        prob = torch.sigmoid(logit)
        dist = Bernoulli(prob)  # p = P(True); 1-p = P(False)
        
        return dist
        
class CriticNetwork(nn.Module):
    """Value Model"""
    def __init__(self, num_inputs, hidden_size):
        super(CriticNetwork, self).__init__()
        
        # Define Critic Network (maybe use Tanh() instead of ReLU() for the hidenlayer)
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        """Runs a forward pass on the NN"""
        value = self.critic(state)
        return value






