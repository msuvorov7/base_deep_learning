import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class PPO(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            gamma: float = 0.99,
            pi_lr: float = 1e-4,
            v_lr: float = 5e-4,
    ):
        super().__init__()

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.pi_optimizer = optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = optim.Adam(self.v_model.parameters(), lr=v_lr)

        self.gamma = gamma

    def get_action(self, state):
        logits = self.pi_model(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.numpy()

    def fit(self, states, actions, rewards, dones, next_states, epoch_n, batch_size: int = 128, epsilon: float = 0.2):
        states, actions, rewards, dones, next_states = map(np.array, [states, actions, rewards, dones, next_states])
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]

        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns, rewards, next_states = map(torch.FloatTensor, [states, actions, returns, rewards, next_states])

        logits = self.pi_model(states)
        dist = torch.distributions.Categorical(logits=logits)

        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(epoch_n):
            idx = np.random.permutation(rewards.shape[0])
            for i in range(0, rewards.shape[0], batch_size):
                b_idx = idx[i: i + batch_size]
                b_states = states[b_idx]
                b_actions = actions[b_idx]
                b_returns = returns[b_idx]
                b_rewards = rewards[b_idx]
                b_next_states = next_states[b_idx]
                b_old_log_probs = old_log_probs[b_idx]

                b_advantage = b_rewards.detach() + self.gamma * self.v_model(b_next_states).detach() - self.v_model(b_states)

                b_logits = self.pi_model(b_states)
                b_dist = torch.distributions.Categorical(logits=b_logits)
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - epsilon,  1. + epsilon) * b_advantage.detach()

                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean((b_returns.detach() - self.v_model(b_states)) ** 2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


def visualize(env, agent, max_len=1000):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = obs

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
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
    env = gym.make('Acrobot-v1')
    env.reset()

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"n_states: {state_dim}, n_actions: {n_actions}")

    episode_n = 30
    trajectory_n = 20
    session_len = 500
    epoch_n = 20
    total_rewards = []

    agent = PPO(state_dim, n_actions)

    for episode in range(episode_n):
        states, actions, rewards, dones, next_states = [], [], [], [], []

        for _ in range(trajectory_n):
            state = env.reset()
            total_reward = 0
            for t in range(session_len):
                states.append(state)

                action = agent.get_action(state)
                actions.append(action)

                next_state, reward, done, _ = env.step(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                state = next_state

                total_reward += reward

                if done:
                    break

            total_rewards.append(total_reward)

        agent.fit(states, actions, rewards, dones, next_states, epoch_n)

        print(f'episode {episode}, mean reward: {np.mean(total_rewards[-trajectory_n:])}')

    visualize(env, agent)

    plt.figure(figsize=(10, 7))
    plt.plot(total_rewards)
    plt.title('Total Rewards')
    plt.grid()
    plt.show()
