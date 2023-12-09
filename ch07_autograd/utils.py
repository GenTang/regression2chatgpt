# -*- coding: UTF-8 -*-
'''
此脚本用于定义Scalar类，以及相应的可视化工具
'''


from graphviz import Digraph
import math


class Scalar:
    
    def __init__(self, value, prevs=[], op=None, label='', requires_grad=True):
        # 节点的值
        self.value = value
        # 节点的标识（label）和对应的运算（op），用于作图
        self.label = label
        self.op = op
        # 节点的前节点，即当前节点是运算的结果，而前节点是参与运算的量
        self.prevs = prevs
        # 是否需要计算该节点偏导数，即∂loss/∂self（loss表示最后的模型损失）
        self.requires_grad = requires_grad
        # 该节点偏导数，即∂loss/∂self
        self.grad = 0.0
        # 如果该节点的prevs非空，存储所有的∂self/∂prev
        self.grad_wrt = dict()
        # 作图需要，实际上对计算没有作用
        self.back_prop = dict()
        
    def __repr__(self):
        return f'Scalar(value={self.value:.2f}, grad={self.grad:.2f})'
    
    def __add__(self, other):
        '''
        定义加法，self + other将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self + other
        output = Scalar(self.value + other.value, [self, other], '+')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = 1
        output.grad_wrt[self] = 1
        # 计算偏导数 ∂output/∂other = 1
        output.grad_wrt[other] = 1
        return output
    
    def __sub__(self, other):
        '''
        定义减法，self - other将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self - other
        output = Scalar(self.value - other.value, [self, other], '-')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = 1
        output.grad_wrt[self] = 1
        # 计算偏导数 ∂output/∂other = -1
        output.grad_wrt[other] = -1
        return output
    
    def __mul__(self, other):
        '''
        定义乘法，self * other将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self * other
        output = Scalar(self.value * other.value, [self, other], '*')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = other
        output.grad_wrt[self] = other.value
        # 计算偏导数 ∂output/∂other = self
        output.grad_wrt[other] = self.value
        return output
    
    def __pow__(self, other):
        '''
        定义乘方，self**other将触发该函数
        '''
        assert isinstance(other, (int, float))
        # output = self ** other
        output = Scalar(self.value ** other, [self], f'^{other}')
        output.requires_grad = self.requires_grad
        # 计算偏导数 ∂output/∂self = other * self**(other-1)
        output.grad_wrt[self] = other * self.value**(other - 1)
        return output
    
    def sigmoid(self):
        '''
        定义sigmoid
        '''
        s = 1 / (1 + math.exp(-1 * self.value))
        output = Scalar(s, [self], 'sigmoid')
        output.requires_grad = self.requires_grad
        # 计算偏导数 ∂output/∂self = output * (1 - output)
        output.grad_wrt[self] = s * (1 - s)
        return output
    
    def __rsub__(self, other):
        '''
        定义右减法，other - self将触发该函数
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        output = Scalar(other.value - self.value, [self, other], '-')
        output.requires_grad = self.requires_grad or other.requires_grad
        # 计算偏导数 ∂output/∂self = -1
        output.grad_wrt[self] = -1
        # 计算偏导数 ∂output/∂other = 1
        output.grad_wrt[other] = 1
        return output
    
    def __radd__(self, other):
        '''
        定义右加法，other + self将触发该函数
        '''
        return self.__add__(other)
    
    def __rmul__(self, other):
        '''
        定义右乘法，other * self将触发该函数
        '''
        return self * other
    
    def backward(self, fn=None):
        '''
        由当前节点出发，求解以当前节点为顶点的计算图中每个节点的偏导数，i.e. ∂self/∂node
        参数
        ----
        fn ：画图函数，如果该变量不等于None，则会返回向后传播每一步的计算的记录
        返回
        ----
        re ：向后传播每一步的计算的记录
        '''
        def _topological_order():
            '''
            利用深度优先算法，返回计算图的拓扑排序（topological sorting）
            '''
            def _add_prevs(node):
                if node not in visited:
                    visited.add(node)
                    for prev in node.prevs:
                        _add_prevs(prev)
                    ordered.append(node)
            ordered, visited = [], set()
            _add_prevs(self)
            return ordered

        def _compute_grad_of_prevs(node):
            '''
            由node节点出发，向后传播
            '''
            # 作图需要，实际上对计算没有作用
            node.back_prop = dict()
            # 得到当前节点在计算图中的梯度。由于一个节点可以在多个计算图中出现，
            # 使用cg_grad记录当前计算图的梯度
            dnode = cg_grad[node]
            # 使用node.grad记录节点的累积梯度
            node.grad += dnode
            for prev in node.prevs:
                # 由于node节点的偏导数已经计算完成，可以向后扩散（反向传播）
                # 需要注意的是，向后扩散到上游节点是累加关系
                grad_spread = dnode * node.grad_wrt[prev]
                cg_grad[prev] = cg_grad.get(prev, 0.0) + grad_spread
                node.back_prop[prev] = node.back_prop.get(prev, 0.0) + grad_spread
        
        # 当前节点的偏导数等于1，因为∂self/∂self = 1。这是反向传播算法的起点
        cg_grad = {self: 1}
        # 为了计算每个节点的偏导数，需要使用拓扑排序的倒序来遍历计算图
        ordered = reversed(_topological_order())
        re = []
        for node in ordered:
            _compute_grad_of_prevs(node)
            # 作图需要，实际上对计算没有作用
            if fn is not None:
                re.append(fn(self, 'backward'))
        return re


def _get_node_attr(node, direction='forward'):
    '''
    节点的属性
    '''
    node_type = _get_node_type(node)
    def _forward_attr():
        if node_type == 'param':
            node_text = f'{{ grad=None | value={node.value: .2f} | {node.label}}}'
            return dict(label=node_text, shape='record', fontsize='10', fillcolor='springgreen', style='filled, bold')
        elif node_type == 'computation':
            node_text = f'{{ grad=None | value={node.value: .2f} | {node.op}}}'
            return dict(label=node_text, shape='record', fontsize='10', fillcolor='gray94', style='filled, rounded')
        elif node_type == 'input':
            if node.label == '':
                node_text = f'input={node.value: .2f}'
            else:
                node_text = f'{node.label}={node.value: .2f}'
            return dict(label=node_text, shape='oval', fontsize='10')
    
    def _backward_attr():
        attr = _forward_attr()
        attr['label'] = attr['label'].replace('grad=None', f'grad={node.grad: .2f}')
        if not node.requires_grad:
            attr['style'] = 'dashed'
        # 为了作图美观
        # 如果向后扩散（反向传播）的梯度等于0，或者扩散给不需要梯度的节点，那么该节点用虚线表示
        grad_back = [v if k.requires_grad else 0 for (k, v) in node.back_prop.items()]
        if len(grad_back) > 0 and sum(grad_back) == 0:
            attr['style'] = 'dashed'
        return attr 
    
    if direction == 'forward':
        return _forward_attr()
    else:
        return _backward_attr()
    
    
def _get_node_type(node):
    '''
    决定节点的类型，计算节点、参数以及输入数据
    '''
    if node.op is not None:
        return 'computation'
    if node.requires_grad:
        return 'param'
    return 'input'


def _trace(root):
    '''
    遍历图中的所有点和边
    '''
    nodes, edges = set(), set()
    def _build(v):
        if v not in nodes:
            nodes.add(v)
            for prev in v.prevs:
                edges.add((prev, v))
                _build(prev)
    _build(root)
    return nodes, edges


def _draw_node(graph, node, direction='forward'):
    '''
    画节点
    '''
    node_attr = _get_node_attr(node, direction)
    uid = str(id(node)) + direction
    graph.node(name=uid, **node_attr)


def _draw_edge(graph, n1, n2, direction='forward'):
    '''
    画边
    '''
    uid1 = str(id(n1)) + direction
    uid2 = str(id(n2)) + direction
    def _draw_back_edge():
        if n1.requires_grad and n2.requires_grad:
            grad = n2.back_prop.get(n1, None)
            if grad is None:
                graph.edge(uid2, uid1, arrowhead='none', color='deepskyblue')   
            elif grad == 0:
                graph.edge(uid2, uid1, style='dashed', label=f'{grad: .2f}', color='deepskyblue')
            else: 
                graph.edge(uid2, uid1, label=f'{grad: .2f}', color='deepskyblue')
        else:
            graph.edge(uid2, uid1, style='dashed', arrowhead='none', color='deepskyblue')

    if direction == 'forward':
        graph.edge(uid1, uid2)
    elif direction == 'backward':
        _draw_back_edge()
    else:
        _draw_back_edge()
        graph.edge(uid1, uid2)


def draw_graph(root, direction='forward'):
    '''
    图形化展示由root为顶点的计算图
    参数
    ----
    root ：Scalar，计算图的顶点
    direction ：str，向前传播（forward）或者反向传播（backward）
    返回
    ----
    re ：Digraph，计算图
    '''
    nodes, edges = _trace(root)
    rankdir = 'BT' if direction == 'forward' else 'TB'
    graph = Digraph(format='svg', graph_attr={'rankdir': rankdir})
    for item in nodes:
        _draw_node(graph, item, direction)
    for n1, n2 in edges:
        _draw_edge(graph, n1, n2, direction)
    return graph
