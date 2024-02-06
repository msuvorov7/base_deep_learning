import random
import time

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
from copy import deepcopy
from collections import deque


class SAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 pi_lr: float = 1e-3,
                 q_lr: float = 1e-3,
                 gamma: float = 0.99,
                 alpha: float = 1e-3,
                 tau: float = 1e-2,
                 capacity: int = 10_000
                 ):
        super(SAC, self).__init__()
        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * action_dim),
            nn.Tanh()
        )

        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.memory = deque([], maxlen=capacity)

        self.pi_optimizer = optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.q1_optimizer = optim.Adam(self.q1_model.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_model.parameters(), lr=q_lr)

    def predict_action(self, states: torch.FloatTensor):
        means, log_stds = self.pi_model(states).T
        means, log_stds = means.unsqueeze(1), log_stds.unsqueeze(1)
        dist = Normal(means, torch.exp(log_stds))
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def get_action(self, state):
        states = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.predict_action(states)
        return action.squeeze(1).detach().numpy()

    def update_model(self, loss, optimizer, model=None, target_model=None):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (model is not None) and (target_model is not None):
            for param, target_param in zip(model.parameters(), target_model.parameters()):
                new_target_param = (1 - self.tau) * target_param + self.tau * param
                target_param.data.copy_(new_target_param)

    def fit(self, state, action, reward, done, next_state, batch_size: int = 64):
        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, [states, actions, rewards, dones, next_states])
            rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

            next_actions, next_log_probs = self.predict_action(next_states)
            next_states_and_actions = torch.concat([next_states, next_actions], dim=1)
            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q1_target_model(next_states_and_actions)
            next_q_values = torch.min(next_q1_values, next_q2_values)
            targets = rewards + self.gamma * (1 - dones) * (next_q_values - self.alpha * next_log_probs)

            states_and_actions = torch.concat([states, actions], dim=1)
            q1_loss = torch.mean((targets.detach() - self.q1_model(states_and_actions)) ** 2)
            q2_loss = torch.mean((targets.detach() - self.q2_model(states_and_actions)) ** 2)
            self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
            self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

            pred_actions, pred_log_probs = self.predict_action(states)
            states_and_pred_actions = torch.concat([states, pred_actions], dim=1)
            pred_q1_values = self.q1_target_model(states_and_pred_actions)
            pred_q2_values = self.q1_target_model(states_and_pred_actions)
            pred_q_values = torch.min(pred_q1_values, pred_q2_values)
            pi_loss = - torch.mean(pred_q_values - self.alpha * pred_log_probs)
            self.update_model(pi_loss, self.pi_optimizer)


def visualize(env, agent, max_len=1000):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = obs

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(2 * action)
        trajectory['rewards'].append(reward)

        state = obs

        time.sleep(0.03)
        env.render()

        if done:
            break

    return trajectory


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"n_states: {state_dim}, n_actions: {action_dim}")

    episode_n = 50
    session_len = 200
    total_rewards = []

    agent = SAC(state_dim, action_dim)

    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        for _ in range(session_len):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(2 * action)
            agent.fit(state, action, reward, done, next_state)

            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

        print(f'episode {episode}, reward: {total_reward}')

    visualize(env, agent)
