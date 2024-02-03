import gym  # 0.24.0
import time
import numpy as np

env = gym.make('Taxi-v3')
env.reset()


class CrossEntropyAgent:
    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.policy = np.ones((self.n_states, self.n_actions), dtype=float) / self.n_actions

    def get_action(self, state: int) -> int:
        """
        Выбор действия агента в зависимости от состояния среды
        :param state: состояние среды
        :return: действие агента
        """
        action = np.random.choice(np.arange(self.n_actions), p=self.policy[state])
        return int(action)

    def generate_session(self, t_max: int = 10_000) -> tuple:
        """
        Генерация сессии игры
        :param t_max: максимальное число шагов
        :return: набор из списка состояний, списка действий и списка наград
        """
        states, actions = [], []
        total_reward = 0.0
        state = env.reset()

        for t in range(t_max):
            action = self.get_action(int(state))

            new_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            total_reward += reward

            state = new_state
            if done:
                break
        return states, actions, total_reward

    def update_policy(self, elite_states: list, elite_actions: list, learning_rate: float = None) -> np.ndarray:
        """
        Обновление политик при найденных элитных траекториях
        :param elite_states: Набор элитных состояний
        :param elite_actions: Набор элитных действий
        :param learning_rate: параметр сглаживания по Лапласу
        :return: Обновленная матрица политик
        """
        new_policy = np.zeros((self.n_states, self.n_actions))

        for ind in range(len(elite_actions)):
            new_policy[elite_states[ind], elite_actions[ind]] += 1

        if learning_rate is None:
            for state in range(self.n_states):
                if np.sum(new_policy[state]) > 0:
                    new_policy[state] /= np.sum(new_policy[state])
                else:
                    new_policy[state] = self.policy[state].copy()
        else:
            new_policy = (new_policy + learning_rate) / (new_policy.sum(axis=1, keepdims=True) + learning_rate * self.n_actions)

        return new_policy

    def fit(self,
            n_sessions: int = 250,
            percentile: int = 50,
            n_iterations: int = 100,
            learning_rate: float = 0.3,
            smooth: str = 'policy'
            ):
        """
        Обучение агента
        :param n_sessions: Число генерируемых сессий
        :param percentile: перцентиль для выбора элитных траекторий
        :param n_iterations: число итераций обучения
        :param learning_rate: параметр сглаживания
        :param smooth: метод сглаживания
        :return:
        """
        if smooth not in ['policy', 'laplace']:
            raise NotImplementedError

        for i in range(n_iterations):
            sessions = [self.generate_session() for _ in range(n_sessions)]
            states_batch, actions_batch, rewards_batch = zip(*sessions)
            elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

            if smooth == 'policy':
                new_policy = self.update_policy(elite_states, elite_actions)
                self.policy = learning_rate * new_policy + (1 - learning_rate) * self.policy
            elif smooth == 'laplace':
                self.policy = self.update_policy(elite_states, elite_actions, learning_rate)
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

        time.sleep(0.3)
        env.render()

        if done:
            break

    return trajectory


if __name__ == '__main__':
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print(f"n_states: {n_states}, n_actions: {n_actions}")
    agent = CrossEntropyAgent(n_states, n_actions)
    agent.fit()
    trajectory = visualize(env, agent, max_len=1000)
