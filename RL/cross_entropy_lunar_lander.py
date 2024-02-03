import gym  # 0.24.0
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('LunarLander-v2')
env.reset()


class Network(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        return self.out(self.relu(self.linear(x)))


class CrossEntropyAgent:
    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.network = Network(n_states, n_actions)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-2)
        self.loss = nn.CrossEntropyLoss()

    def get_action(self, state: np.ndarray, eps: float = 0.0) -> int:
        """
        Выбор действия агента в зависимости от состояния среды
        :param state: состояние среды
        :return: действие агента
        """
        state = torch.FloatTensor(state)
        logits = self.network(state)
        probs = (1 - eps) * F.softmax(logits, dim=-1).detach().numpy() + eps * np.ones(self.n_actions) / self.n_actions
        action = np.random.choice(np.arange(self.n_actions), p=probs / probs.sum())
        return int(action)

    def generate_session(self, t_max: int = 10_000, eps: float = 0.0) -> tuple:
        """
        Генерация сессии игры
        :param t_max: максимальное число шагов
        :param eps: epsilon
        :return: набор из списка состояний, списка действий и списка наград
        """
        states, actions = [], []
        total_reward = 0.0
        state = env.reset()

        for t in range(t_max):
            action = self.get_action(state, eps=eps)

            new_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            total_reward += reward

            state = new_state
            if done:
                break
        return states, actions, total_reward

    def update_policy(self, elite_states: list, elite_actions: list):
        """
        Обновление политик при найденных элитных траекториях
        :param elite_states: Набор элитных состояний
        :param elite_actions: Набор элитных действий
        :return: Обновленная матрица политик
        """
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))

        loss = self.loss(self.network(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit(self,
            n_sessions: int = 250,
            session_len: int = 1_000,
            eps: float = 1.0,
            percentile: int = 50,
            n_iterations: int = 100
            ):
        """
        Обучение агента
        :param n_sessions: Число генерируемых сессий
        :param session_len: Длина сессии
        :param eps: epsilon
        :param percentile: перцентиль для выбора элитных траекторий
        :param n_iterations: число итераций обучения
        :return:
        """
        for i in range(n_iterations):
            sessions = [self.generate_session(session_len, eps / (i + 1)) for _ in range(n_sessions)]
            states_batch, actions_batch, rewards_batch = zip(*sessions)
            elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

            self.update_policy(elite_states, elite_actions)
            print(f'iteration {i}: mean reward {np.mean(rewards_batch)}')


def select_elites(states_batch: list, actions_batch: list, rewards_batch: list, percentile: int = 50) -> tuple:
    """
    Выбор элитных траекторий
    :param states_batch: список траекторий
    :param actions_batch: список действий
    :param rewards_batch: список наград
    :param percentile: перцентиль для выбора
    :return: Набор из элитных траекорий и элитных действий
    """
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = []
    elite_actions = []

    for state, action, reward in zip(states_batch, actions_batch, rewards_batch):
        if reward >= reward_threshold:
            elite_states += state
            elite_actions += action

    return elite_states, elite_actions


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
    n_states = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"n_states: {n_states}, n_actions: {n_actions}")
    agent = CrossEntropyAgent(8, 4)
    agent.fit(n_sessions=100, session_len=200, eps=0, percentile=50, n_iterations=100)
    trajectory = visualize(env, agent, max_len=1000)

