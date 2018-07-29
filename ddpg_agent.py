import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import random
import gym
from collections import deque
import utils
from math import sqrt

class Actor(nn.Module):
    def __init__(self, state_num, action_num, hidden_unit=16):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden_unit)
        self.fc1.weight.data.uniform_(0, 1/sqrt(state_num))
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.fc2.weight.data.uniform_(0, 1/sqrt(hidden_unit))
        
        self.out = nn.Linear(hidden_unit, action_num)
        self.out.weight.data.uniform_(0, 0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.out(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_num, action_num, hidden_unit=16):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_num + action_num, hidden_unit)
        self.fc1.weight.data.uniform_(0, 1/sqrt(state_num+action_num))
        self.fc2 = nn.Linear(hidden_unit,hidden_unit)
        self.fc2.weight.data.uniform_(0, 1/sqrt(hidden_unit))
        
        self.out = nn.Linear(hidden_unit, 1)
        self.out.weight.data.uniform_(0, 1/sqrt(hidden_unit))
        
    def forward(self, state, action):
        
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        x = F.relu(self.fc2(x))
        return self.out(x)


MemoryBuffer = namedtuple(
    'MemoryBuffer', ['state', 'action', 'reward', 'next_state', 'done'])


class DDPGAgent(object):
    def __init__(self, state_num, action_num, \
        action_range=(-1.,1.), buffer_size=2000, batch_size=64, gamma=0.99, \
        device=torch.device('cpu'), hidden_unit=16):

        self.device = device

        # create A-C Networks
        self.actor = Actor(state_num, action_num, hidden_unit=hidden_unit).to(device)
        self.target_actor = Actor(state_num, action_num, hidden_unit=hidden_unit).to(device)
        self.critic = Critic(state_num, action_num, hidden_unit=hidden_unit).to(device)
        self.target_critic = Critic(state_num, action_num, hidden_unit=hidden_unit).to(device)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

        # add optimize
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-3)

        # add other params
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_buffer = deque(maxlen=buffer_size)
        self.noise = utils.OrnsteinUhlenbeckActionNoise(action_num)
        self.action_range = action_range
        self.critic_loss_F = nn.MSELoss()


    def step(self, state, action, reward, next_state, done):
        self._add(state, action, reward, next_state, done)
        if (len(self.memory_buffer) < self.batch_size):
            return
        self._learn()

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor.forward(state).detach()
        new_action = action.cpu().data.numpy() + self.noise.sample()
        return np.clip(new_action, self.action_range[0], self.action_range[1])

    def _add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = MemoryBuffer(state, action, reward, next_state, done)
        self.memory_buffer.append(e)

    def _sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory_buffer, k=self.batch_size)
        device = self.device

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def _learn(self):

        states, actions, rewards, next_states, dones = self._sample()

        # optimize Actor Network
        loss_actor = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # optimize Critic Network
        next_actions = self.target_actor(next_states)
        y_expected = rewards + (1.0 - dones) * self.gamma * \
            self.target_critic(next_states, next_actions.detach())
        y_predicted = self.critic(states, actions)
        loss_critic = self.critic_loss_F(y_predicted, y_expected.detach())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # soft update Target Networks
        utils.soft_update(self.target_actor, self.actor, 0.01)
        utils.soft_update(self.target_critic, self.critic, 0.01)

    def save_model_params(self, dir, i_eps):
        torch.save(self.critic.state_dict(),'%s/critic_state_dict_%d.pkl' % (dir, i_eps))
        torch.save(self.actor.state_dict(),'%s/actor_state_dict_%d.pkl' % (dir, i_eps))

    def load_model_params(self, dir, i_eps):
        self.critic.load_state_dict(torch.load('%s/critic_state_dict_%d.pkl' % (dir, i_eps)))
        self.actor.load_state_dict(torch.load('%s/actor_state_dict_%d.pkl' % (dir, i_eps)))

        utils.hard_update(self.target_critic, self.critic)
        utils.hard_update(self.target_actor, self.actor)

    def play_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor.forward(state).detach()
        return np.clip(action.cpu().data.numpy(), -self.action_limit, self.action_limit)
