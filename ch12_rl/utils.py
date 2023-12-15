# -*- coding: UTF-8 -*-
'''
此脚本用于定义游戏以及相应的可视化工具
'''


import matplotlib.pyplot as plt
import torch
import pandas as pd


class Lottery:
    
    def __init__(self):
        # 定义游戏的两个状态
        self.params = {
            'w': (1, 1),
            'l': (-1, 1)
        }
    
    def reset(self):
        self.state = 'w' if torch.randn(1).item() > 0 else 'l'
        return self.state
        
    def step(self, action):
        # 如果状态是t，则终止游戏
        if self.state == 't':
            return self.state, 0
        # 1表示抽奖; 0表示终止
        center, std = self.params[self.state]
        if action == 0:
            self.state = 't'
            return 't', 0
        else:
            reward = torch.normal(center, std, (1,)).item()
        # 有10%的概率终止游戏
        if torch.rand(1).item() < 0.01:
            self.state = 't'
        return self.state, reward


def plot_values(v):
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=100)
    v = pd.DataFrame(v)
    for k in v:
        v[k].plot(label=k, legend=True)
    legend = plt.legend(shadow=True, loc='best', fontsize=20)
    plt.yticks(range(-10, 11, 4))
    return fig


def plot_action_probs(v):
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=100)
    v = pd.DataFrame(v)
    for k in v:
        # 在图中画出抽奖的概率
        v[k].apply(lambda x: x[1]).plot(label=k, legend=True)
    legend = plt.legend(shadow=True, loc='best', fontsize=20)
    return fig