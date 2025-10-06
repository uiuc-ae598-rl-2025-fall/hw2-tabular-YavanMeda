import gymnasium as gym
import numpy as np

def initialize_env(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q_array = np.zeros((n_states, n_actions))
    pi = np.ones((n_states, n_actions)) / n_actions
    returns = np.empty((n_states, n_actions), dtype=object)
    for s in range(n_states):
        for a in range(n_actions):
            returns[s][a] = []
    return pi, Q_array, returns, n_states, n_actions

def choose_action(state, pi, n_actions):
    actions = np.arange(n_actions)
    return np.random.choice(actions, p = pi[state])

def generate_episode(env, pi):
    state, _ = env.reset()
    episode = []
    terminated = False
    while not terminated:
        action = choose_action(state, pi, env.action_space.n)
        s_prime, reward, terminated, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = s_prime
    return episode
    
def mc_control(episode, Q, returns, pi, gamma, epsilon, n_actions):
    G = 0
    visited = np.zeros_like(Q, dtype=bool)
    for i in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[i]
        G = gamma*G + reward
        if not (visited[state][action]):
            visited[state][action] = True
            returns[state][action].append(G)
            Q[state, action] = np.average(returns[state][action])
            A_star = np.argmax(Q[state])
            for a in range(n_actions):
                if a==A_star:
                    pi[state][a] = 1 - epsilon + (epsilon / n_actions)
                else:
                    pi[state][a] = epsilon / n_actions
    return Q_array

gamma = 0.95
epsilon = 0.5

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4",  is_slippery=True)
pi, Q_array, returns, n_states, n_actions = initialize_env(env)

n_episodes=10000
for i in range(n_episodes):
    episode = generate_episode(env, pi)
    Q_array = mc_control(episode, Q_array, returns, pi, gamma, epsilon, n_actions)
    

V = np.max(Q_array, axis = 1).reshape(env.unwrapped.desc.shape)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,5))
sns.heatmap(V, annot=True, cmap="viridis", cbar=True)
plt.title(f"State Value Function (V), {n_episodes}")
plt.show()


