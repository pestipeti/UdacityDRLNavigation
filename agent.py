#  MIT License
#
#  Copyright (c) 2020 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import torch
import numpy as np
import random

import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
from model import NavigationModel

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NavigationAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size) -> None:
        """Initialize a NavigationAgent.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0

        # Q-Network
        self.net_local = NavigationModel(state_size, action_size, fc1_units=64, fc2_units=32).to(device)
        self.net_target = NavigationModel(state_size, action_size, fc1_units=64, fc2_units=32).to(device)
        self.optimizer = optim.Adam(self.net_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    def step(self, state, action, reward, next_state, done):

        # Add experience to memory.
        self.memory.add(state, action, reward, next_state, done)

        # Learn after every UPDATE_EVERY timesteps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            # Learn only if we have enough samples in our memory.
            if len(self.memory) > BATCH_SIZE:
                self.learn(self.memory.sample(), gamma=GAMMA)

    def act(self, state, eps=0.):
        """Returns an action for the given state.

        Args:
            state (np.ndarray): current state
            eps (float): epsilon for epsilon-greedy action selection

        Returns:
            (int) action-index
        """
        if random.random() > eps:

            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            self.net_local.eval()
            with torch.no_grad():
                action_values = self.net_local(state)
            self.net_local.train()

            return np.argmax(action_values.cpu().data.numpy())

        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma=0.99):
        """Updates value parameters using given batch of past experience.

        Args:
            experiences (Tuple[torch.tensor]): tuple of (s, a, r, s', next)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next state) from the target model.
        Q_targets_next = self.net_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for the current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model.
        Q_expected = self.net_local(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for target_param, local_param in zip(self.net_target.parameters(), self.net_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

    def load(self, filename):
        self.net_local.load_state_dict(torch.load(filename))

    def save(self, filename):
        torch.save(self.net_local.state_dict(), filename)


# From Udacity Deep Reinforcement Learning nanodegree.
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
