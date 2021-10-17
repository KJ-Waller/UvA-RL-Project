import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Simply pass (obs/state) through the network
        actions = self.l2(F.relu(self.l1(state)))
        
        return actions