#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:50:29 2024

@author: Ermanno Sannini, Daniele Bifolco
"""

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Support function
def plot_policy(policy):

    def get_Z(player_hand, dealer_showing, usable_ace):
        if (player_hand, dealer_showing, usable_ace) in policy:
            return policy[player_hand, dealer_showing, usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(1, 11)
        y_range = np.arange(11, 22)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace) for dealer_showing in x_range] for player_hand in range(21, 10, -1)])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 2), vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])
        plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        plt.yticks(y_range)
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Hand')
        ax.grid(color='black', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
        cbar.ax.invert_yaxis() 
            
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace', fontsize=16)
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace', fontsize=16)
    get_figure(False, ax)
    plt.show()

# def plot_value_function(V, title="Value Function"):
#     """
#     Plots the value function as a surface plot.
#     """
#     min_x = min(k[0] for k in V.keys())
#     max_x = max(k[0] for k in V.keys())
#     min_y = min(k[1] for k in V.keys())
#     max_y = max(k[1] for k in V.keys())

#     x_range = np.arange(min_x, max_x + 1)
#     y_range = np.arange(min_y, max_y + 1)
#     X, Y = np.meshgrid(x_range, y_range)

#     # Find value for all (x, y) coordinates
#     Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
#     Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

#     def plot_surface(X, Y, Z, title):
#         fig = plt.figure(figsize=(16,8))
#         ax = fig.add_subplot(111, projection='3d')
#         surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                                cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
#         ax.set_xlabel('Player Sum')
#         ax.set_ylabel('Dealer Showing')
#         ax.set_zlabel('Value')
#         ax.set_title(title)
#         ax.view_init(ax.elev, -120)
#         fig.colorbar(surf)
#         plt.show()

#     plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
#     plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
def create_epsilon_greedy_action_policy(env,Q,epsilon):
    """ Create epsilon greedy action policy
    Args:
        env: Environment
        Q: Q table
        epsilon: Probability of selecting random action instead of the 'optimal' action
    
    Returns:
        Epsilon-greedy-action Policy function with Probabilities of each action for each state
    """
    def policy(obs):
        P = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n  #initiate with same prob for all actions
        best_action = np.argmax(Q[obs])  #get best action
        P[best_action] += (1.0 - epsilon)
        return P
    return policy

# BJ env 
env = gym.make('Blackjack-v0')
print("Observation space: "+str(env.observation_space))
print("Action space: "+str(env.action_space.n))

#Example Scenario
print("Example scenario")
print(env.reset())
visible = "" if env._get_obs()[2] else "no"
print("The above shows that the player's hand has a total sum of " 
      + str(env._get_obs()[0]) + " while the dealer visible hand is " 
      + str(env._get_obs()[1]) + " and that the player has " + visible + " usable ace")
print(env.step(1))
print(env.step(0))
print(env.dealer)

#Basic naive strategy
def draw_till_17_pol(obs):
    return [1,0] if obs[0]<17 else [0,1]

def calc_payoffs(env,rounds,players,pol):
    """
    Calculate Payoffs.
    
    Args:
        env: environment
        rounds: Number of rounds a player would play
        players: Number of players 
        pol: Policy used
        
    Returns:
        Average payoff
    """
    average_payouts = []
    for player in range(players):
        rd = 1
        total_payout = 0 # to store total payout over 'num_rounds'

        while rd <= rounds:
            action = np.argmax(pol(env._get_obs()))
            obs, payout, is_done, _ = env.step(action)
            if is_done:
                total_payout += payout
                env.reset() # Environment deals new cards to player and dealer
                rd += 1
        average_payouts.append(total_payout)

    plt.plot(average_payouts)                
    plt.xlabel('num_player')
    plt.ylabel('payout after ' + str(rounds) + 'rounds')
    plt.show()    
    print ("Average payout of a player after {} rounds is {}".format(rounds, sum(average_payouts)/players))

print("Basic Naive stategy")
env.reset()
calc_payoffs(env,1000,1000,draw_till_17_pol)
    
#Monte Carlo
def On_pol_mc_control_learn(env, episodes, discount_factor, epsilon):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: Environment.
        episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state to action values.
        Policy is the trained policy that returns action probabilities
    """
    # Keeps track of sum and count of returns for each state
    # An array could be used to save all returns but that's memory inefficient.
    # defaultdict used so that the default value is stated if the observation(key) is not found
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    pol = create_epsilon_greedy_action_policy(env,Q,epsilon)
    
    for i in range(1, episodes + 1):
        # Print out which episode we're on
        if i% 1000 == 0:
            print("\rEpisode {}/{}.".format(i, episodes), end="")
            clear_output(wait=True)

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = pol(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            #First Visit MC:
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    
    return Q, pol

print("Monte Carlo method")
env.reset()
Q_on_pol,On_MC_Learned_Policy = On_pol_mc_control_learn(env, 500000, 0.9, 0.05)

# V = defaultdict(float)
# for state, actions in Q_on_pol.items():
#     action_value = np.max(actions)
#     V[state] = action_value
# plot_value_function(V, title="Optimal Value Function for On-Policy Learning")

#Payoff for On-Policy MC Trained Policy
env.reset()
calc_payoffs(env,1000,1000,On_MC_Learned_Policy)

on_pol = {key: np.argmax(On_MC_Learned_Policy(key)) for key in Q_on_pol.keys()}
print("On-Policy MC Learning Policy")
plot_policy(on_pol)

#TD LEarning
#Sarsa
# def create_epsilon_greedy_action_policy(env,Q,epsilon):
#     """ Create epsilon greedy action policy
#     Args:
#         env: Environment
#         Q: Q table
#         epsilon: Probability of selecting random action instead of the 'optimal' action
    
#     Returns:
#         Epsilon-greedy-action Policy function with Probabilities of each action for each state
#     """
#     def policy(obs):
#         P = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n  #initiate with same prob for all actions
#         best_action = np.argmax(Q[obs])  #get best action
#         P[best_action] += (1.0 - epsilon)
#         return P
#     return policy
def SARSA(env, episodes, epsilon, alpha, gamma):
    """
    SARSA Learning Method
    
    Args:
        env: OpenAI gym environment.
        episodes: Number of episodes to sample.
        epsilon: Probability of selecting random action instead of the 'optimal' action
        alpha: Learning Rate
        gamma: Gamma discount factor
        
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. 
    """
    
    # Initialise a dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The policy we're following
    pol = create_epsilon_greedy_action_policy(env,Q,epsilon)
    for i in range(1, episodes + 1):
        # Print out which episode we're on
        if i% 1000 == 0:
            print("\rEpisode {}/{}.".format(i, episodes), end="")
            clear_output(wait=True)
        curr_state = env.reset()
        probs = pol(curr_state)   #get epsilon greedy policy
        curr_act = np.random.choice(np.arange(len(probs)), p=probs)
        while True:
            next_state,reward,done,_ = env.step(curr_act)
            next_probs = create_epsilon_greedy_action_policy(env,Q,epsilon)(next_state)
            next_act = np.random.choice(np.arange(len(next_probs)),p=next_probs)
            td_target = reward + gamma * Q[next_state][curr_act]
            td_error = td_target - Q[curr_state][curr_act]
            Q[curr_state][curr_act] = Q[curr_state][curr_act] + alpha * td_error
            if done:
                break
            curr_state = next_state
            curr_act = next_act
    return Q, pol

print("SARSA")
env.reset()
Q_SARSA,SARSA_Policy = SARSA(env, 500000, 0.1, 0.1,0.95)

#Payoff for Off-Policy MC Trained Policy
env.reset()
print("SARSA Learning Policy")
calc_payoffs(env,1000,1000,SARSA_Policy)

pol_sarsa = {key: np.argmax(SARSA_Policy(key)) for key in Q_SARSA.keys()}

plot_policy(pol_sarsa)

#Q-Learning
def off_pol_TD_Q_learn(env, episodes, epsilon, alpha, gamma):
    """
    Off-Policy TD Q-Learning Method
    
    Args:
        env: OpenAI gym environment.
        episodes: Number of episodes to sample.
        epsilon: Probability of selecting random action instead of the 'optimal' action
        alpha: Learning Rate
        gamma: Gamma discount factor
        
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. 
    """
    # Initialise a dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The policy we're following
    pol = create_epsilon_greedy_action_policy(env,Q,epsilon)
    for i in range(1, episodes + 1):
        if i% 1000 == 0:
            print("\rEpisode {}/{}.".format(i, episodes), end="")
            clear_output(wait=True)
        curr_state = env.reset()
        while True:
            probs = pol(curr_state)   #get epsilon greedy policy
            curr_act = np.random.choice(np.arange(len(probs)), p=probs)
            next_state,reward,done,_ = env.step(curr_act)
            next_act = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][next_act]
            td_error = td_target - Q[curr_state][curr_act]
            Q[curr_state][curr_act] = Q[curr_state][curr_act] + alpha * td_error
            if done:
                break
            curr_state = next_state
    return Q, pol

print("Q-Learning")
env.reset()
Q_QLearn,QLearn_Policy = off_pol_TD_Q_learn(env, 500000, 0.1, 0.1,0.95)
#Payoff for Off-Policy Q-Learning Trained Policy
env.reset()
calc_payoffs(env,1000,1000,QLearn_Policy)

pol_QLearn = {key: np.argmax(QLearn_Policy(key)) for key in Q_QLearn.keys()}
print("Off-Policy Q Learning Policy")
plot_policy(pol_QLearn)