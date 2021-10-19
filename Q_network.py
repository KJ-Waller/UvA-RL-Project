import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128]):
        nn.Module.__init__(self)
        self.l1 = nn.Sequential()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.l1.add_module(f'lin{i}', nn.Linear(state_dim, hidden_dim))
            else:
                self.l1.add_module(f'lin{i}', nn.Linear(prev_hidden_dim, hidden_dim))
            prev_hidden_dim = hidden_dim
            self.l1.add_module(f'activ{i}', nn.LeakyReLU())

        self.l2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Simply pass (obs/state) through the network
        actions = F.softmax(self.l2(F.relu(self.l1(state))))
        
        return actions

    def save_model(self, directory, fname):
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.state_dict(), os.path.join(directory, f'{fname}.pth'))