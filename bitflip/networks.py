import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    """
    """
    def __init__(self, n_actions, name, input_dims, chkpt_dir, lr=0.001):
        """
        """
        super(DQN).__init__()
        self.chpt_dir = chkpt_dir
        self.chpt_file = os.path.join(self.chpt_dir, name)

        self.fc1 = self.linear(input_dims, 256)
        self.fc2 = self.linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) 

    def forward(self, state):
        """
        """
        f1 = F.relu(self.fc1(state))
        return self.fc2(f1)
    
    def save_checkpoint(self):
        """
        """
        print('saving')
        T.save(self.state_dict(), self.chpt_file)
    
    def load_checkpoint(self):
        """
        """
        print("loading")
        self.load_state_dict(T.load(self.chpt_file))

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        actions = self.fc2(flat1)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))