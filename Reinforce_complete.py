#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE on the switched–actions corridor.
Two models:
(a) Action-only features x(s,left)=[0,1]^T, x(s,right)=[1,0]^T (same for all s).
(b) Tabular features, one preference per (s,a).
Both use REINFORCE with a state-value baseline.
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Define environment

class CorridorSwitchedEnv:
    """Corridor environment with switched actions
    Two actions: 0 (left) and 1 (right)
    States 0, 1, 2 and 3 (terminal)
    
    Returns:
        state:     current state
        reward     current reward
        terminal   whether state is terminal
    """
    
    def __init__(self):
        self.nb_states = 3          # non-terminal states: 0,1,2
        self.nb_actions = 2
        self.state = 0
        
    def reset(self):
        """Reset the MDP to its initial state"""
        self.state = 0
        return 0
    
    def step(self, action):
        """Take a step in the MDP"""
        terminal = False
        if self.state == 0:
            # from state 0, action 1 moves right to state 1
            if action == 1:
                self.state = 1
        elif self.state == 1:
            # actions are swapped in this state
            if action == 0:        # "right"
                self.state = 2
            else:                  # "left"
                self.state = 0
        elif self.state == 2:
            if action == 0:        # "left"
                self.state = 1
            else:                  # "right" to goal
                self.state = 3
                terminal = True
                
        return self.state, -1, terminal
    

def generate_episode(env, policy, pi):
    """Generate one complete episode.
    Returns:
        trajectory: list of (state, action, reward) for t=0,...,T-1
        T:          trajectory length
    """
    trajectory = []
    done = False
    s = env.reset()
    while not done:
        a = policy(s, pi)
        s_next, r, done = env.step(a)
        trajectory.append((s, a, r))
        s = s_next
    T = len(trajectory)
    return trajectory, T

#%% Define value function and policies

class TabularStateValueFunction:
    """Tabular state-value function v_hat(s; psi)"""
    def __init__(self, learning_rate, nb_states):
        self.learning_rate = learning_rate
        self.psi = np.zeros(nb_states)
    
    def evaluate(self, state):
        """Compute value in given state"""
        return self.psi[state]
    
    def train(self, state, target):
        """One-step SGD update towards target return"""
        self.psi[state] += self.learning_rate * (target - self.psi[state])


class ActionOnlyPolicy:
    """
    Linear softmax policy with action-only features.
    Features follow the slide:
        x(s,left)  = [0,1]^T
        x(s,right) = [1,0]^T
    for all states s, i.e. the policy does not depend on s.
    We implement this as a single preference vector theta in R^2.
    """
    def __init__(self, learning_rate, nb_actions, init_theta):
        self.learning_rate = learning_rate
        self.nb_actions = nb_actions
        self.theta = init_theta.astype(float)  # shape (2,)
    
    def pi(self, state):
        """Softmax over action preferences (independent of state)."""
        prefs = self.theta - np.max(self.theta)
        exp_prefs = np.exp(prefs)
        return exp_prefs / np.sum(exp_prefs)
    
    def update(self, state, action, advantage):
        """
        REINFORCE update:
            theta <- theta + alpha * A_t * grad_theta log pi(a_t|s_t)
        grad for softmax:
            grad_j log pi(a|s) = 1[a=j] - pi(j|s)
        """
        probs = self.pi(state)
        grad = -probs
        grad[action] += 1.0
        self.theta += self.learning_rate * advantage * grad


class TabularPolicy:
    """Tabular softmax policy with separate preferences per (s,a)."""
    def __init__(self, learning_rate, nb_states, nb_actions, init_theta):
        self.learning_rate = learning_rate
        self.nb_actions = nb_actions
        # theta[s,a] is the preference of action a in state s
        self.theta = np.outer(np.ones(nb_states), init_theta.astype(float))
    
    def pi(self, state):
        """Returns probability distribution over actions at given state"""
        prefs = self.theta[state] - np.max(self.theta[state])
        exp_prefs = np.exp(prefs)
        return exp_prefs / np.sum(exp_prefs)
    
    def update(self, state, action, advantage):
        """REINFORCE update at (state,action) for a given advantage."""
        probs = self.pi(state)
        grad = -probs
        grad[action] += 1.0
        self.theta[state] += self.learning_rate * advantage * grad

        
#%% Run environment: REINFORCE with baseline for two models

# repeats for averaging
repeats = 20
# number of episodes to run
episodes = 1000
# discount factor
gamma = 1.0
# learning rate for state-value function
alpha_psi = 2**-6
# learning rate for policy
alpha_theta = 2**-9
# initial policy weights
init_theta = np.array([0.0, 0.0])

env = CorridorSwitchedEnv()

def behaviour_policy(st, pi):
    """Sample an action from the current policy object."""
    return np.random.choice(range(env.nb_actions), p=pi.pi(st))

# Average reward obtained after a given number of training episodes
hist_R_action = np.zeros(episodes)   # model (a): action-only features
hist_R_tabular = np.zeros(episodes)  # model (b): tabular (s,a) features

for rep in range(repeats):
    # -------- Model (a): action-only features --------
    v_hat_a = TabularStateValueFunction(learning_rate=alpha_psi,
                                        nb_states=env.nb_states)
    pi_a = ActionOnlyPolicy(learning_rate=alpha_theta,
                            nb_actions=env.nb_actions,
                            init_theta=init_theta)
    
    for ep in range(episodes):
        traj, T = generate_episode(env, behaviour_policy, pi_a)
        # Total reward of the episode
        G_total = sum(r for (_, _, r) in traj)
        hist_R_action[ep] += G_total
        
        # REINFORCE with state-value baseline
        G = 0.0
        for (s, a, r) in reversed(traj):
            G = r + gamma * G
            baseline = v_hat_a.evaluate(s)
            advantage = G - baseline
            v_hat_a.train(s, G)
            pi_a.update(s, a, advantage)
    
    # -------- Model (b): tabular state-action features --------
    v_hat_b = TabularStateValueFunction(learning_rate=alpha_psi,
                                        nb_states=env.nb_states)
    pi_b = TabularPolicy(learning_rate=alpha_theta,
                         nb_states=env.nb_states,
                         nb_actions=env.nb_actions,
                         init_theta=init_theta)
    
    for ep in range(episodes):
        traj, T = generate_episode(env, behaviour_policy, pi_b)
        G_total = sum(r for (_, _, r) in traj)
        hist_R_tabular[ep] += G_total
        
        G = 0.0
        for (s, a, r) in reversed(traj):
            G = r + gamma * G
            baseline = v_hat_b.evaluate(s)
            advantage = G - baseline
            v_hat_b.train(s, G)
            pi_b.update(s, a, advantage)

# Average across repeats
hist_R_action /= repeats
hist_R_tabular /= repeats

#%% Plot results

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(hist_R_action, linewidth=1.5,
        label='Model (a): action-only features')
ax.plot(hist_R_tabular, linewidth=1.5,
        label='Model (b): tabular $(s,a)$ features')
ax.set_ylabel('Total reward per episode', fontsize=12)
ax.set_xlabel('Episode', fontsize=12)
ax.set_title('REINFORCE Algorithm: Comparison of Two Feature Representations', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reinforce_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Training Complete ===")
print(f"Final average reward (last 100 episodes):")
print(f"  Model (a) action-only:  {np.mean(hist_R_action[-100:]):.2f}")
print(f"  Model (b) tabular:      {np.mean(hist_R_tabular[-100:]):.2f}")

