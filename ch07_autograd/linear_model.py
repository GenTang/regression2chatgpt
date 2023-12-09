# -*- coding: UTF-8 -*-
'''
此脚本用于定义线性回归模型
'''


from utils import Scalar


def mse(errors):
    '''
    计算均方误差
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
        定义线性回归模型的参数：a, b
        '''
        self.a = Scalar(0.0, label='a')
        self.b = Scalar(0.0, label='b')

    def forward(self, x):
        '''
        根据当前的参数估计值，得到模型的预测结果
        '''
        return self.a * x + self.b
    
    def error(self, x, y):
        '''
        当前数据的模型误差
        '''
        return y - self.forward(x)

    def string(self):
        '''
        输出当前模型的结果
        '''
        return f'y = {self.a.value:.2f} * x + {self.b.value:.2f}'
