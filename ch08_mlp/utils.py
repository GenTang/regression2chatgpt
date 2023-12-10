# -*- coding: UTF-8 -*-
'''
定义多层感知器的模型组件，比如线性模型，Sigmoid函数等
'''


import torch
import torch.nn.functional as F
import numpy as np


class Linear:
    
    def __init__(self, in_features, out_features, bias=True):
        '''
        模型参数初始化
        需要注意的是，此次故意没做参数初始化的优化
        '''
        self.weight = torch.randn((in_features, out_features))
        self.bias = torch.randn(out_features) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        '''
        返回线性模型的参数，主要用于参数迭代更新
        由于PyTorch的计算单元就是张量，
        所以此次只需将不同参数简单合并成列表即可
        '''
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


class Sigmoid:
    
    def __call__(self, x):
        self.out = torch.sigmoid(x)
        return self.out
    
    def parameters(self):
        '''
        Sigmoid函数没有模型参数
        '''
        return []


class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        '''
        Tanh函数没有模型参数
        '''
        return []


class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        '''
        将各层的模型参数简单合并成列表即可
        '''
        return [p for layer in self.layers for p in layer.parameters()]
    
    def predict_proba(self, x):
        '''
        为了数据可视化，计算模型输出的概率
        '''
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        logits = self(x)
        self.proba = F.softmax(logits, dim=1).detach().numpy()
        return self.proba
