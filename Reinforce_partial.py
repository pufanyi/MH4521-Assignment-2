#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:27:02 2024

"""

#%% Import libraries

import numpy as np
import matplotlib.pyplot as plt

#%% Define environment

class CorridorSwitchedEnv:
    """Corridor environment with switched actions
    Two actions: 0 and 1
    States 0, 1, 2 and 3 (terminal)
    
    Returns:
        state:     current state
        reward     current reward
        terminal   whether state is terminal
    """
    
    def __init__(self):
        self.nb_states = 3
        self.nb_actions = 2
        self.state = 0
        
    def reset(self):
        """Reset the MPD to its initial state"""
        self.state = 0
        return 0
    
    def step(self, action):
        """Take a step in the MDP"""
        terminal = False
        if self.state == 0:
            if action == 1:
                self.state = 1
        elif self.state == 1:
            # Swap left and right in this state
            if action == 0:
                self.state = 2
            else:
                self.state = 0
        elif self.state == 2:
            if action == 0:
                self.state = 1
            else:
                self.state = 3
                terminal = True
                
        return self.state, -1, terminal
    

def generate_episode(env, policy, pi):
    """Generate one complete episode.
    
    Returns:
        trajectory: list of tuples [(state, reward, not_done, action), ...],
        T:          trajectory length (not counting terminal state)
    """
    trajectory = []
    done = False
    t = 0
    At = 0
    while (t == 0) or not(done):
        if t == 0:
            St = env.reset()
            Rt = None
        else:
            St, Rt, done = env.step(At)
        if not(done):
            At = policy(St, pi)
        else:
            At = None
        trajectory.append((St, Rt, done, At))
        t += 1
    T = t-1
    
    return trajectory, T

#%% Define policy and value approximations

class TabularStateValueFunction:
    """Tabular state-value function"""
    def __init__(self, learning_rate, nb_states):
        self.learning_rate = learning_rate
        self.psi = np.zeros(nb_states)
    
    def evaluate(self, state):
        """Compute value in given state"""
        # [...]
    
    def train(self, state, target):
        """Train at state for observed target"""
        # [...]


class TabularPolicy:
    """Tabular action-state function"""
    def __init__(self, learning_rate, nb_states, nb_actions, init_theta):
        self.learning_rate = learning_rate
        self.nb_actions = nb_actions
        self.theta = np.outer(np.ones(nb_states), init_theta)
    
    def pi(self):
        """Returns probability distribution over actions at given state"""
        # [...]

    def update(self):
        """Update policy at state-action pair for a given discounted return"""
        # [...]
        
#%% Run environment

# REINFORCE with baseline

# repeats
repeats = 1
# number of episodes to run
episodes = 1000
# discount factor
gamma = 1.0
# learning rate for state-value function
alpha_psi = 2**-6
# learning rate for policy
alpha_theta = 2**-9
# initial policy weights
init_theta = np.array([0, 0])

env = CorridorSwitchedEnv()

def policy(st, pi):
    return np.random.choice(range(env.nb_actions), p=pi.pi(st))

# Average reward obtained after a given number of training episodes
hist_R = np.zeros(episodes)

for rep in range(repeats):
    # Instantiate tabular state-value function
    v_hat = TabularStateValueFunction(learning_rate=alpha_psi, nb_states=env.nb_states)

    # Instantiate tabular policy
    pi = TabularPolicy(learning_rate=alpha_theta,
                              nb_states=env.nb_states,
                              nb_actions=env.nb_actions,
                              init_theta=init_theta)

    for ep in range(episodes):

        traj, T = generate_episode(env, policy, pi)
        # trajectory is of the form
        # [(state, reward, done, action), ...]

        # [...]

hist_R = hist_R/repeats

#%% Save results

hist_R_base = hist_R
hist_R_nobase = hist_R

#%% Plot results

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(hist_R_base, linewidth=1.0, color='lime',
        label='REINFORCE with baseline $\\alpha_\\theta=2^{-9}$ $\\alpha_\psi=2^{-6}$')
ax.plot(hist_R_nobase, linewidth=1.0, color='red', label='REINFORCE $\\alpha=2^{-13}$')
ax.set_ylabel('Total reward per episode')
ax.set_xlabel('Episode')
ax.legend()
