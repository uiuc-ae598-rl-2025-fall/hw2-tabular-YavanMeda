import mc, sarsa, q_learn, gymnasium as gym, numpy as np, matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4",  is_slippery=True)

n_episodes=6000
gamma = 0.95
epsilon = 0.1
alpha = 0.1
runs = 20

def greedy_policy(Q_array):
    n_states, n_actions = Q_array.shape
    pi_greedy = np.zeros((n_states,), dtype=int)
    for state in range(n_states):
        pi_greedy[state] = np.argmax(Q_array[state])
    return pi_greedy

def evaluate_policy(env, pi_greedy, gamma, n_eval_episodes):
    eval_returns = []
    for i in range(n_eval_episodes):
        state, _ = env.reset()
        terminated = False
        G = 0.0
        t = 0
        while not terminated:
            action = pi_greedy[state]
            s_prime, reward, terminated, _, _ = env.step(action)
            G += gamma ** t * reward
            t += 1
            state = s_prime
        eval_returns.append(G)
    return np.mean(eval_returns)


def mc_func(env, n_episodes, gamma, epsilon):
    pi, Q_array, returns, n_states, n_actions = mc.initialize_env(env)
    eval_returns_mc = []
    timestep_mc = []
    for i in range(n_episodes):
        episode = mc.generate_episode(env, pi)
        Q_array = mc.mc_control(episode, Q_array, returns, pi, gamma, epsilon, n_actions)
        if (i % 100 == 0):
            pi_greedy = greedy_policy(Q_array)
            avg_returns = evaluate_policy(env, pi_greedy, gamma, 100)
            eval_returns_mc.append(avg_returns)
            timestep_mc.append(i)
    return np.array(eval_returns_mc), np.array(timestep_mc)


def sarsa_func(env, n_episodes, alpha, gamma, epsilon):
    Q_array, n_states, n_actions = sarsa.initialize_env(env)   
    eval_returns_sarsa = []
    timestep_sarsa = []
    for i in range(n_episodes):
        Q_array, G, steps = sarsa.sarsa(env, Q_array, alpha, epsilon, gamma, n_actions)
        if (i % 100 == 0):
            pi_greedy = greedy_policy(Q_array)
            avg_returns = evaluate_policy(env, pi_greedy, gamma, 100)
            eval_returns_sarsa.append(avg_returns)
            timestep_sarsa.append(i)
    return np.array(eval_returns_sarsa), np.array(timestep_sarsa)


def q_learn_func(env, n_episodes, alpha, gamma, epsilon):
    Q_array, n_states, n_actions = q_learn.initialize_env(env)
    eval_returns_q = []
    timestep_q = []
    for i in range(n_episodes):
        Q_array, G, steps = q_learn.q_learn(env, Q_array, alpha, epsilon, gamma, n_actions)
        if (i % 100 == 0):
            pi_greedy = greedy_policy(Q_array)
            avg_returns = evaluate_policy(env, pi_greedy, gamma, 100)
            eval_returns_q.append(avg_returns)
            timestep_q.append(i)
    return np.array(eval_returns_q), np.array(timestep_q)


def average_over_func(func, *args):
    all_returns = []
    timesteps = None
    for run in range(runs):
        returns, timestep = func(env, n_episodes, *args)
        all_returns.append(returns)
        if timesteps is None:
            timesteps = timestep
        if run % 1 == 0:
            print(f"Run {run}/{runs} completed")
    return np.mean(all_returns, axis=0), np.std(all_returns, axis=0), timesteps

#avg_mc, std_mc, timestep_mc = average_over_func(mc_func, gamma, epsilon)
avg_sarsa, std_sarsa, timestep_sarsa = average_over_func(sarsa_func, alpha, gamma, epsilon)
avg_q, std_q, timestep_q = average_over_func(q_learn_func, alpha, gamma, epsilon)

episodes = timestep_sarsa

plt.figure(figsize=(9,5))

def plot_mean_std(mean, std, label, color):
    plt.plot(episodes, mean, label=label, color=color)
    plt.fill_between(episodes, mean-std, mean+std, color=color, alpha = 0.2)

#plot_mean_std(avg_mc, std_mc, "Monte Carlo", "tab:blue")
plot_mean_std(avg_sarsa, std_sarsa, "SARSA", "tab:orange")
plot_mean_std(avg_q, std_q, "Q-Learning", "tab:green")

plt.xlabel("Training Episodes")
plt.ylabel("Evaluation Returns")
plt.title("Evaluation Returns over Environment Timesteps")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()