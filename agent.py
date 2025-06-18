import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # Increased memory size
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # Slower decay for more exploration
        self.learning_rate = 0.001

        # The model and optimizer are created only once
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        # A simple feed-forward neural network
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),  # Increased layer size
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        # Stores an experience in the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # Decides an action using an epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():  # More efficient for inference
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        # Trains the network using a batch of experiences from the replay buffer
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Unpack the batch
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch])
        rewards = torch.FloatTensor([e[2] for e in minibatch])
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.BoolTensor([e[4] for e in minibatch])

        # Get Q-values for current states
        q_values = self.model(states)
        # Select the Q-value for the action that was actually taken
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max Q-values for next states
        with torch.no_grad():
            next_q_values = self.model(next_states)
        max_next_q = next_q_values.max(1)[0]

        # Compute the target Q-value
        target_q = rewards + (self.gamma * max_next_q * ~dones)

        # Compute loss and perform backpropagation
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay