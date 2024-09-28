import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.dynamic_plot import ContActionPlotter


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.episode_rewards = []
        self.episode_values = []
        self.episode_dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               self.episode_rewards,\
               self.episode_values,\
               self.episode_dones,\
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def store_ep(self, ep_rewards):
        self.episode_rewards.append(ep_rewards)

    def store_val(self, values):
        self.episode_values.append(values)

    def store_dones(self, dones):
        self.episode_dones.append(dones)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_dones = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, layers):
        super().__init__()

        self.input_layer = nn.Linear(*input_dims, layers[0])
        self.h1 = nn.Linear(layers[0], layers[1])
        self.mu = nn.Linear(layers[1], n_actions)
        # self.var = nn.Linear(layers[1], n_actions)

        # f = 0.1
        # T.nn.init.orthogonal_(self.input_layer.weight.data, f)
        # T.nn.init.orthogonal_(self.h1.weight.data, f)
        # T.nn.init.orthogonal_(self.mu.weight.data, f)
        # T.nn.init.orthogonal_(self.var.weight.data, f)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.input_layer(state)
        x = T.relu(x)
        x = self.h1(x)
        x = T.relu(x)
        mu = self.mu(x)
        mu = T.tanh(mu)
        # var = self.var(x)
        # var = T.sigmoid(var)
        # var = var*var
        # var = T.sigmoid(var) * 0.1
        return mu#, var


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, layers):
        super().__init__()
        self.input = nn.Linear(*input_dims, layers[0])
        self.hidden = nn.Linear(layers[0], layers[1])
        self.output = nn.Linear(layers[1], 1)

        # f = 0.1
        # T.nn.init.orthogonal_(self.input.weight.data, f)
        # T.nn.init.orthogonal_(self.hidden.weight.data, f)
        # T.nn.init.orthogonal_(self.output.weight.data, f)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.input(state)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        value = self.output(x)
        return value


class Agent:
    def __init__(self, input_dims, n_actions, layers,
                 gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=4):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(n_actions, input_dims, alpha, layers)
        self.critic = CriticNetwork(input_dims, alpha, layers)
        self.memory = PPOMemory(batch_size)
        self.cov_var = T.full(size=(n_actions,), fill_value=0.5, device=self.actor.device)
        self.cov_mat = T.diag(self.cov_var)

        # self.interactive_actor = ContActionPlotter(n_actions)

    def store(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def store_ep(self, rewards):
        self.memory.store_ep(rewards)

    def store_val(self, values):
        self.memory.store_val(values)

    def store_dones(self, dones):
        self.memory.store_dones(dones)


    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # mu, sigma = self.actor(state)
        mu = self.actor(state)

        # dist = Normal(mu, sigma)
        dist = MultivariateNormal(mu, self.cov_mat)
        # self.interactive_actor.update_data(mu.cpu().detach().numpy().flatten(), sigma.cpu().detach().numpy().flatten())
        value = self.critic(state)
        action = dist.sample()
        # action = T.clamp(action, -1.0, 1.0)
        probs = T.squeeze(dist.log_prob(action)).cpu().detach().numpy()
        action = T.squeeze(action).cpu().numpy()
        value = T.squeeze(value).item()

        return action, probs, value

    def compute_discount_rewards(self, ep_rewards_arr):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(ep_rewards_arr):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = np.array(batch_rtgs)

        return batch_rtgs

    def compute_gae_rewards(self, rewards, values, dones, lamb, gamma):
        batch_gae = []
        eps_len = len(rewards)

        for batch_id in range(eps_len):
            gae = 0.0
            r_batch = rewards[eps_len-batch_id-1]
            v_batch = values[eps_len-batch_id-1]
            d_batch = dones[eps_len-batch_id-1]
            batch_len = len(r_batch)
            for i in range(batch_len):
                r = r_batch[batch_len-i-1]
                v = v_batch[batch_len-i-1]
                v_next = 0.0
                if i > 0:
                    v_next = v_batch[batch_len-i]
                d = d_batch[batch_len-i-1]
                mask = int(1-d)

                delta = r + gamma * v_next * mask - v
                gae = delta + gamma * lamb * mask * gae
                batch_gae.insert(0, gae+v)

        return np.array(batch_gae)

    def learn(self, values_hist=None, adv_hist=None):
        state_arr, action_arr, old_probs_arr, vals_arr, \
        reward_arr, done_arr, ep_rewards_arr, ep_values_err, ep_dones_arr, batches = self.memory.generate_batches()
        disc_r_arr = self.compute_discount_rewards(ep_rewards_arr)
        # gae_r_arr = self.compute_gae_rewards(ep_rewards_arr, ep_values_err, ep_dones_arr, self.gae_lambda, self.gamma)

        # advantage = gae_r_arr - vals_arr
        advantage = disc_r_arr - vals_arr
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        if values_hist is not None:
            values_hist.append(vals_arr.mean())
        if adv_hist is not None:
            adv_hist.append(advantage.mean())

        advantage = T.tensor(advantage).to(self.actor.device)
        returns_arr = T.tensor(disc_r_arr).to(self.actor.device)
        for _ in range(self.n_epochs):
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)#.reshape(len(batch), 1)
                # mu, sigma = self.actor(states)
                mu = self.actor(states)
                dist = MultivariateNormal(mu, self.cov_mat)
                # dist = Normal(mu, sigma)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                # prob_ratio = prob_ratio.mean(dim=-1)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[
                    batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = returns_arr[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                entropy = -dist.entropy().mean()

                total_loss = actor_loss + 1.0 * critic_loss + 0.1 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()









def main():
    env = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
    N = 2048
    batch_size = 256
    n_epochs = 10
    alpha = 0.0003
    state_space = env.observation_space.shape
    action_space = env.action_space.shape[0]
    layers = [200, 120]
    agent = Agent(n_actions=action_space, batch_size=batch_size, input_dims=state_space,
                  layers=layers, alpha=alpha, n_epochs=n_epochs)
    need_render = False
    episodes = 3000
    max_steps = 2000
    scores, values, adv = learn_policy_problem(env, agent, episodes, max_steps, N, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)
    # agent.interactive_actor.close()

def learn_policy_problem(env, agent, episodes, max_steps, N, need_render):
    scores = []
    j = 0
    values = []
    adv = []
    for e in range(episodes):
        state, _ = env.reset()
        score = 0
        ep_rewards = []
        ep_values = []
        ep_dones = []

        for i in range(max_steps):
            if need_render:
                env.render()
            action, prob, val = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action.flatten())
            score += reward
            ep_rewards.append(reward)
            ep_values.append(val)
            ep_dones.append(done)
            j += 1
            agent.store(state, action, prob, val, reward, done)
            state = next_state
            if done:
                break
        agent.store_ep(ep_rewards)
        agent.store_val(ep_values)
        agent.store_dones(ep_dones)
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
        if j >= N:
            j = 0
            print("learn...")
            agent.learn(values, adv)

    return scores, values, adv

def result_learning(env, agent, max_steps):
    agent.epsilon = 0.0
    while True:
        state, _ = env.reset()
        score = 0
        for i in range(max_steps):
            env.render()
            action, _, _ = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action.flatten())
            score += reward
            state = next_state
            if done:
                break
        print("score: {}".format(score))

if __name__ == '__main__':
    main()