import gym  # 0.24.0
from gym.core import ObservationWrapper
import numpy as np
import time

from collections import defaultdict


class QLearningAgent:
    def __init__(self, alpha: float, gamma: float, epsilon: float, get_legal_actions: callable):
        self.get_legal_actions = get_legal_actions
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """ Returns Q(state,action) """
        return self.q_values[state][action]

    def set_q_value(self, state, action, value):
        """ Sets the Q_value for [state,action] to the given value """
        self.q_values[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q_values
        V(s) = max_over_action Q(state,action) over possible actions.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value = max([
            self.get_q_value(state, action) for action in possible_actions
        ])

        return value

    def update(self, state, action, reward, next_state):
        """
        Q-Value update:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.gamma
        lr = self.alpha

        new_value = (1 - lr) * self.get_q_value(state, action) + lr * (reward + gamma * self.get_value(next_state))

        self.set_q_value(state, action, new_value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        best_action = max(
            possible_actions,
            key=lambda action: self.get_q_value(state, action)
        )

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action


class SARSAAgent(QLearningAgent):
    def update(self, state, action, reward, next_state):
        """
        Q-Value update:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.gamma
        lr = self.alpha

        next_action = self.get_action(next_state)
        new_value = (1 - lr) * self.get_q_value(state, action) + lr * (reward + gamma * self.get_q_value(next_state, next_action))

        self.set_q_value(state, action, new_value)


class MonteCarloAgent:
    def __init__(self, gamma: float, epsilon: float, get_legal_actions: callable):
        self.get_legal_actions = get_legal_actions
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0))
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_values = defaultdict(lambda: defaultdict(lambda: 0))

    def get_q_value(self, state, action):
        """ Returns Q(state,action) """
        return self.q_values[state][action]

    def get_n_value(self, state, action):
        """ Returns Q(state,action) """
        return self.n_values[state][action]

    def update(self, states, actions, rewards):
        """
        Q-Value update:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.gamma

        G = np.zeros(len(rewards))
        G[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            G[t] = rewards[t] + gamma * G[t + 1]

        for t in range(len(rewards)):
            self.q_values[states[t]][actions[t]] += (G[t] - self.q_values[states[t]][actions[t]]) / (self.n_values[states[t]][actions[t]] + 1)
            self.n_values[states[t]][actions[t]] += 1

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        best_action = max(
            possible_actions,
            key=lambda action: self.get_q_value(state, action)
        )

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action


class Binarizer(ObservationWrapper):
    def observation(self, state):
        n_digits = [1, 1, 1, 1]
        state = [round(x, n) for x, n in zip(state, n_digits)]
        return tuple(state)


def fit_q_learning_agent(env, agent, episode_n: int, session_len: int):
    total_rewards = []
    for episode in range(episode_n):
        total_reward = 0.0
        state = env.reset()

        for _ in range(session_len):
            # get agent to pick action given state
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            # train (update) agent for state
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            if done:
                break

        total_rewards.append(total_reward)

        if episode % 100 == 0:
            print(f'iteration: {episode}, eps: {agent.epsilon:.5f}, mean reward: {np.mean(total_rewards[-10:]):.1f}')

        agent.epsilon = 1 - (episode / episode_n)

    return total_rewards


def fit_monte_carlo_agent(env, agent, episode_n: int, session_len: int):
    total_rewards = []
    for episode in range(episode_n):
        state = env.reset()
        states, actions, rewards = [], [], []

        for _ in range(session_len):
            states.append(state)

            # get agent to pick action given state
            action = agent.get_action(state)
            actions.append(action)

            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        total_rewards.append(sum(rewards))
        # train (update) agent for state
        agent.update(states, actions, rewards)

        if episode % 100 == 0:
            print(f'iteration: {episode}, eps: {agent.epsilon:.5f}, mean reward: {np.mean(total_rewards[-10:]):.1f}')

        agent.epsilon = 1 - (episode / episode_n)

    return total_rewards


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
    env = Binarizer(gym.make('CartPole-v1'))
    env.reset()

    n_states = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"n_states: {n_states}, n_actions: {n_actions}")

    q_learning_agent = QLearningAgent(
        alpha=0.5,
        gamma=0.99999,
        epsilon=1,
        get_legal_actions=lambda s: range(n_actions)
    )

    q_learning_rewards = fit_q_learning_agent(env, q_learning_agent, episode_n=10_000, session_len=500)

    sarsa_agent = SARSAAgent(
        alpha=0.5,
        gamma=0.99999,
        epsilon=1,
        get_legal_actions=lambda s: range(n_actions)
    )

    sarsa_rewards = fit_q_learning_agent(env, sarsa_agent, episode_n=10_000, session_len=500)

    monte_carlo_agent = MonteCarloAgent(
        gamma=0.999999,
        epsilon=1,
        get_legal_actions=lambda s: range(n_actions)
    )

    monte_carlo_rewards = fit_monte_carlo_agent(env, monte_carlo_agent, episode_n=10_000, session_len=10_000)
    sarsa_agent.epsilon = 0
    visualize(env, sarsa_agent)
