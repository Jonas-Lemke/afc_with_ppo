"""
This file contains the actor-critic network for the PPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

# import torch.optim as optim
# from torch.distributions.categorical import Categorical


class ActorCriticNetwork(nn.Module):
    """Policy and Value Model"""
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCriticNetwork, self).__init__()

        # Define Actor Network (maybe use Tanh() instead of ReLU() for the hidenlayer)
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        
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
        logit = self.actor(state)
        
        # # add noise for exploration
        # logit += 0.5 * torch.randn_like(logit)
        
        # # clip to prevent fully determenistic logits (e.g. always 0 or 1)
        # logit = torch.clamp(logit, -10, 10)

        prob = torch.sigmoid(logit)
        # prob = torch.softmax(logits, dim=-1)
        
        dist = Bernoulli(prob)  # p = P(True); 1-p = P(False)

        value = self.critic(state)
        
        return dist, value





