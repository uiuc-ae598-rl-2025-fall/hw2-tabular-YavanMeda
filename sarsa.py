import gymnasium as gym
import numpy as np

def initialize_env(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q_array = np.zeros((n_states, n_actions))
    return Q_array, n_states, n_actions

def choose_action(state, Q_array, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q_array[state])


def sarsa(env, Q_array, alpha, epsilon, gamma, n_actions):
    total_reward = 0
    t=0
    state, _ = env.reset()
    action = choose_action(state, Q_array, n_actions, epsilon)
    terminated = False
    while not terminated:
        s_prime, reward, terminated, _, _ = env.step(action)
        a_prime = choose_action(s_prime, Q_array, n_actions, epsilon)
        Q_array[state][action] = Q_array[state][action]+ alpha*(reward + gamma*Q_array[s_prime][a_prime] - Q_array[state][action])
        state = s_prime
        action = a_prime
        total_reward += reward
        t += 1
    return Q_array, total_reward, t


gamma = 0.95
epsilon = 0.5
alpha = 0.1

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4",  is_slippery=False)
Q_array, n_states, n_actions = initialize_env(env)

n_episodes = 10000
for i in range(n_episodes):
    Q_array, _, _ = sarsa(env, Q_array, alpha, epsilon, gamma, n_actions)

# V = np.max(Q_array, axis = 1).reshape(env.unwrapped.desc.shape)

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(5,5))
# sns.heatmap(V, annot=True, cmap="viridis", cbar=True)
# plt.title(f"State Value Function (V), {n_episodes}")
# plt.show()