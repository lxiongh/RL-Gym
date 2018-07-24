import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import random
import gym
from collections import deque
from math import sqrt
import utils

class Net(nn.Module):
    def __init__(self, state_num, action_num,hidden_unit=16):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden_unit)
        self.fc1.weight.data.uniform_(0, 1/sqrt(state_num))
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.fc2.weight.data.uniform_(0, 1/sqrt(hidden_unit))
        
        self.out = nn.Linear(hidden_unit, action_num)
        self.out.weight.data.uniform_(0, 1/sqrt(hidden_unit))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

ReplayMemory = namedtuple('ReplayMemory', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNAgent(object):
    def __init__(self, state_num, action_num, \
        eps=0.1, buffer_size=1000000, batch_size=128, gamma=0.99, \
        device=torch.device("cpu"), hidden_unit=64, lr=1e-3):

        self.policy_net = Net(state_num, action_num, hidden_unit=hidden_unit).to(device)
        self.target_net = Net(state_num, action_num, hidden_unit=hidden_unit).to(device)
        self.replay_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory = deque(maxlen=buffer_size)
        self.update_delay = 100
        self.t_step = 0
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.eps = eps

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
    
    def step(self, state, action, reward, next_state, done):
        self._add(state, action, reward, next_state, done)
        if (len(self.replay_memory) < self.batch_size):
            return
        self._learn()

    def choose_action(self, state):
        if np.random.uniform() < self.eps:
            # random choose
            return np.random.randint(0, self.action_num)
        else:
            # by policy net
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action = self.policy_net(state).max(0)[1].cpu().data.item()
            return action
    
    def choose_opt_action(self, state):
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action = self.policy_net(state).max(0)[1].cpu().data.item()
            return action
            
    def _add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = ReplayMemory(state, action, reward, next_state, done)
        self.replay_memory.append(e)
    
    def _sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.replay_memory, k=self.batch_size)
        device = self.device

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def _learn(self):
        
        if (self.t_step % self.update_delay == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.t_step = (self.t_step + 1) % self.update_delay

        experiences = self._sample()
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model_params(self, dir, i_eps):
        torch.save(self.policy_net.state_dict(),'%s/policy_state_dict_%d.pkl' % (dir, i_eps))

    def load_model_params(self, dir, i_eps):
        params = torch.load('%s/policy_state_dict_%d.pkl' % (dir, i_eps))
        self.policy_net.load_state_dict(params)
        
        utils.hard_update(self.target_net, self.policy_net)

    def play_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.policy_net.forward(state).detach()
        return action.cpu().data.item()
