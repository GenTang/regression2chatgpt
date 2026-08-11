# -*- coding: UTF-8 -*-
'''
Define multilayer-perceptron components such as linear layers and sigmoid functions
'''


import torch
import torch.nn.functional as F
import numpy as np


class Linear:
    
    def __init__(self, in_features, out_features, bias=True):
        '''
        Initialize model parameters
        Note that parameter initialization is intentionally left unoptimized here
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
        Return the linear-layer parameters for iterative updates
        Because tensors are PyTorch's computational units,
        Therefore, simply combine the different parameters into a list
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
        The sigmoid function has no model parameters
        '''
        return []


class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        '''
        The tanh function has no model parameters
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
        Combine the model parameters from all layers into a list
        '''
        return [p for layer in self.layers for p in layer.parameters()]
    
    def predict_proba(self, x):
        '''
        Calculate model-output probabilities for visualization
        '''
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        logits = self(x)
        self.proba = F.softmax(logits, dim=1).detach().numpy()
        return self.proba
