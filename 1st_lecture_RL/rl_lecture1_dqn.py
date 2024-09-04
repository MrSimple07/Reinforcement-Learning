import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import numpy.random as rnd

from collections import deque
import numpy as np
import random


class DenseNetwork(nn.Module):
    def __init__(self, state_space, action_space, layers, policy):
        nn.Module.__init__(self)
        self.input = nn.Linear(state_space, layers[0])
        self.ls = len(layers)
        if self.ls > 1:
            self.l1 = nn.Linear(layers[0], layers[1])
        if self.ls > 2:
            self.l2 = nn.Linear(layers[1], layers[2])
        self.output = nn.Linear(layers[-1], action_space)
        if policy:
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
            self.loss = F.mse_loss

    def forward(self, x):
        x = F.relu(self.input(x))
        if self.ls > 1:
            x = F.relu(self.l1(x))
        if self.ls > 2:
            x = F.relu(self.l2(x))
        x = self.output(x)
        return x


def build_dense(state_space, action_space, layers, policy=False):
    return DenseNetwork(state_space, action_space, layers, policy)


class Memory:
    def __init__(self, max_len):
        self.memory = deque(maxlen=max_len)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.memory)


class PytorchDqnAgent():
    def __init__(self, state_space, action_space, layers, gamma=0.99, tau=0.95, memory_size=20000, batch_size=32):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.95
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.memory = Memory(memory_size)
        self.layers = layers

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # self.device = torch.device('cpu')
        self.policy_network = build_dense(self.state_space, self.action_space, self.layers, policy=True)
        self.target_network = build_dense(self.state_space, self.action_space, self.layers)
        self.policy_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()

    def act(self, state):
        return self.epsi_policy(state)

    def epsi_policy(self, state):
        if rnd.random() <= self.epsilon:
            act = rnd.randint(self.action_space)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            act_values = self.policy_network(state_tensor)
            act = torch.argmax(act_values).item()
        return act

    def replay(self):
        if self.memory.size() < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device).view(self.batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        predict_y = self.policy_network(states).gather(1, actions)
        target_y = self.target_network(next_states).max(1)[0].detach()
        target_y = rewards + self.gamma * target_y * (1 - dones)
        loss = self.policy_network.loss(predict_y, target_y.unsqueeze(1))

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def update_target(self):
        weights = self.policy_network.state_dict()
        target_weights = self.target_network.state_dict()
        for k in weights.keys():
            target_weights[k] = (1 - self.tau) * target_weights[k] + self.tau * (weights[k])
        self.target_network.load_state_dict(weights)

    def end_episode(self):
        self.update_target()
        self.decay()


def learn_problem(env, agent, episodes, max_steps, need_render):
    scores = []

    for e in range(episodes):
        state, info = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action = agent.act(state)

            next_state, reward, done, _, _ = env.step(action)
            score += reward
            agent.memory.remember(state, action, reward, next_state, int(done))
            state = next_state
            agent.replay()
            if done:
                break
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)

    return scores


def main():
    env = gym.make('CartPole-v1', render_mode="human")
    state_space = 4
    action_space = 2
    layers = [32, 20]
    agent = PytorchDqnAgent(state_space, action_space, layers)
    need_render = True
    episodes = 500
    max_steps = 500
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
