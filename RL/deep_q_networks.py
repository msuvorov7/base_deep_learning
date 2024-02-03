import random
import time
from collections import deque

import torch
import torch.nn as nn

import numpy as np
import gym  # 0.24.0


class QFunction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QFunction, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

    def forward(self, x):
        return self.model(x)


class DQN:
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99, capacity: int = 10_000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.q_function = QFunction(self.state_dim, self.action_dim)
        self.target_function = QFunction(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.memory = deque([], maxlen=capacity)

    def get_action(self, state, epsilon: float):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def fit(self, state, action, reward, done, next_state, batch_size: int, iteration):
        """
        y = reward + gamma * max(Q_func(next_state, next_action))
        Loss = MSE(y, Q_func(state, action))
        Q_func <- Q_func - lr * grad(Loss)
        """
        self.memory.append([state, action, reward, int(done), next_state])
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            actions = actions.type('torch.LongTensor')

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(batch_size), actions]

            loss = self.loss(targets.detach(), q_values)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


class HardTargetDQN(DQN):
    """
    Set Q_target = Q_func
    Do a lot of iterations:
        y = reward + gamma * max(Q_target(next_state, next_action))
        Loss = MSE(y, Q_func(state, action))
        Q_func <- Q_func - lr * grad(Loss)
    Q_target <- Q_func
    """
    def fit(self, state, action, reward, done, next_state, batch_size: int, iteration):
        self.memory.append([state, action, reward, int(done), next_state])
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            actions = actions.type('torch.LongTensor')

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.target_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(batch_size), actions]

            loss = self.loss(targets.detach(), q_values)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.q_function.parameters(), 100)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (iteration % 100) == 0:
                self.target_function.load_state_dict(self.q_function.state_dict())


class SoftTargetDQN(DQN):
    def fit(self, state, action, reward, done, next_state, batch_size: int, iteration):
        """
        y = reward + gamma * max(Q_target(next_state, next_action))
        Loss = MSE(y, Q_func(state, action))
        Q_func <- Q_func - lr * grad(Loss)
        Q_target <- tau * Q_func + (1-tau) * Q_target
        """
        self.memory.append([state, action, reward, int(done), next_state])
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            actions = actions.type('torch.LongTensor')

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.target_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(batch_size), actions]

            loss = self.loss(targets.detach(), q_values)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.q_function.parameters(), 100)
            self.optimizer.step()
            self.optimizer.zero_grad()

            target_net_state_dict = self.target_function.state_dict()
            policy_net_state_dict = self.q_function.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * 0.01 + target_net_state_dict[key] * (1 - 0.01)
            self.target_function.load_state_dict(target_net_state_dict)


class DoubleDQN(DQN):
    def fit(self, state, action, reward, done, next_state, batch_size: int, iteration):
        """
        y = reward + gamma * Q_func(next_state, argmax(Q_target(next_state, next_action)) )
        Loss = MSE(y, Q_func(state, action))
        Q_func <- Q_func - lr * grad(Loss)
        Q_target <- tau * Q_func + (1-tau) * Q_target
        """
        self.memory.append([state, action, reward, int(done), next_state])
        if len(self.memory) > batch_size:
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            actions = actions.type('torch.LongTensor')

            next_actions = torch.argmax(self.q_function(next_states), dim=1)
            best_next_q_value = self.target_function(next_states)[torch.arange(batch_size), next_actions]
            targets = rewards + self.gamma * (1 - dones) * best_next_q_value

            q_values = self.q_function(states)[torch.arange(batch_size), actions]

            loss = self.loss(targets.detach(), q_values)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.q_function.parameters(), 100)
            self.optimizer.step()
            self.optimizer.zero_grad()

            target_net_state_dict = self.target_function.state_dict()
            policy_net_state_dict = self.q_function.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * 0.01 + target_net_state_dict[key] * (1 - 0.01)
            self.target_function.load_state_dict(target_net_state_dict)


def visualize(env, agent, max_len=1000):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = obs

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state, 0)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = obs

        time.sleep(0.03)
        env.render()

        if done:
            break

    return trajectory


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"n_states: {state_dim}, n_actions: {action_dim}")

    agent = DoubleDQN(state_dim, action_dim)
    episode_n = 200
    session_len = 1000
    epsilon = 1
    total_rewards = []
    iteration = 0

    for episode in range(episode_n):
        total_reward = 0

        state = env.reset()
        for _ in range(session_len):
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            agent.fit(state, action, reward, done, next_state, 64, iteration)
            iteration += 1

            state = next_state

            if done:
                break

        epsilon = 1 - (episode / episode_n)**0.25
        total_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f'iteration: {episode}, eps: {epsilon:.5f}, mean reward: {np.mean(total_rewards[-10:]):.1f}')

    visualize(env, agent)
