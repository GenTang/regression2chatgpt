# -*- coding: UTF-8 -*-
'''
This script defines the game and its visualization utilities
'''


import matplotlib.pyplot as plt
import torch
import pandas as pd


class Lottery:
    
    def __init__(self):
        # Define the game's two states
        self.params = {
            'w': (1, 1),
            'l': (-1, 1)
        }
    
    def reset(self):
        self.state = 'w' if torch.randn(1).item() > 0 else 'l'
        return self.state
        
    def step(self, action):
        # Stop the game if the state is t
        if self.state == 't':
            return self.state, 0
        # 1 means draw; 0 means stop
        center, std = self.params[self.state]
        if action == 0:
            self.state = 't'
            return 't', 0
        else:
            reward = torch.normal(center, std, (1,)).item()
        # There is a 10% probability of stopping the game
        if torch.rand(1).item() < 0.01:
            self.state = 't'
        return self.state, reward


def plot_values(v):
    # Configure a font for displaying Chinese text in Matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # Display negative signs correctly
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    # Create a figure
    fig = plt.figure(figsize=(6, 6), dpi=100)
    v = pd.DataFrame(v)
    for k in v:
        v[k].plot(label=k, legend=True)
    legend = plt.legend(shadow=True, loc='best', fontsize=20)
    plt.yticks(range(-10, 11, 4))
    return fig


def plot_action_probs(v):
    # Configure a font for displaying Chinese text in Matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # Display negative signs correctly
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    # Create a figure
    fig = plt.figure(figsize=(6, 6), dpi=100)
    v = pd.DataFrame(v)
    for k in v:
        # Plot the probability of drawing
        v[k].apply(lambda x: x[1]).plot(label=k, legend=True)
    legend = plt.legend(shadow=True, loc='best', fontsize=20)
    return fig