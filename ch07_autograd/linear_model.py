# -*- coding: UTF-8 -*-
'''
This script defines a linear regression model
'''


from utils import Scalar


def mse(errors):
    '''
    Calculate Mean Squared Error
    '''
    n = len(errors)
    wrt = {}
    value = 0.0
    requires_grad = False
    for item in errors:
        value += item.value ** 2 / n
        wrt[item] = 2 / n * item.value
        requires_grad = requires_grad or item.requires_grad
    output = Scalar(value, errors, 'mse')
    output.requires_grad=requires_grad
    output.grad_wrt = wrt
    return output


class Linear:
    
    def __init__(self):
        '''
        Define the linear regression model parameters: a, b
        '''
        self.a = Scalar(0.0, label='a')
        self.b = Scalar(0.0, label='b')

    def forward(self, x):
        '''
        Calculate model predictions using the current parameter estimates
        '''
        return self.a * x + self.b
    
    def error(self, x, y):
        '''
        Model error on the current data
        '''
        return y - self.forward(x)

    def string(self):
        '''
        Output the current model results
        '''
        return f'y = {self.a.value:.2f} * x + {self.b.value:.2f}'
