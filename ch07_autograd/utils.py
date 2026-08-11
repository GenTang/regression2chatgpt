# -*- coding: UTF-8 -*-
'''
This script defines the Scalar class and its visualization utilities
'''


from graphviz import Digraph
import math


class Scalar:
    
    def __init__(self, value, prevs=[], op=None, label='', requires_grad=True):
        # Node value
        self.value = value
        # Node label and corresponding operation (op), used for plotting
        self.label = label
        self.op = op
        # Predecessors of the node: the current node is the result, while its predecessors are the operands
        self.prevs = prevs
        # Whether to calculate this node's partial derivative ∂loss/∂self, where loss is the final model loss
        self.requires_grad = requires_grad
        # The node's partial derivative ∂loss/∂self
        self.grad = 0.0
        # If prevs is not empty, store every partial derivative ∂self/∂prev
        self.grad_wrt = dict()
        # Required for plotting but has no effect on the calculation
        self.back_prop = dict()
        
    def __repr__(self):
        return f'Scalar(value={self.value:.2f}, grad={self.grad:.2f})'
    
    def __add__(self, other):
        '''
        Define addition; self + other invokes this method
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self + other
        output = Scalar(self.value + other.value, [self, other], '+')
        output.requires_grad = self.requires_grad or other.requires_grad
        # Calculate the partial derivative ∂output/∂self = 1
        output.grad_wrt[self] = 1
        # Calculate the partial derivative ∂output/∂other = 1
        output.grad_wrt[other] = 1
        return output
    
    def __sub__(self, other):
        '''
        Define subtraction; self - other invokes this method
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self - other
        output = Scalar(self.value - other.value, [self, other], '-')
        output.requires_grad = self.requires_grad or other.requires_grad
        # Calculate the partial derivative ∂output/∂self = 1
        output.grad_wrt[self] = 1
        # Calculate the partial derivative ∂output/∂other = -1
        output.grad_wrt[other] = -1
        return output
    
    def __mul__(self, other):
        '''
        Define multiplication; self * other invokes this method
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        # output = self * other
        output = Scalar(self.value * other.value, [self, other], '*')
        output.requires_grad = self.requires_grad or other.requires_grad
        # Calculate the partial derivative ∂output/∂self = other
        output.grad_wrt[self] = other.value
        # Calculate the partial derivative ∂output/∂other = self
        output.grad_wrt[other] = self.value
        return output
    
    def __pow__(self, other):
        '''
        Define exponentiation; self**other invokes this method
        '''
        assert isinstance(other, (int, float))
        # output = self ** other
        output = Scalar(self.value ** other, [self], f'^{other}')
        output.requires_grad = self.requires_grad
        # Calculate the partial derivative ∂output/∂self = other * self**(other-1)
        output.grad_wrt[self] = other * self.value**(other - 1)
        return output
    
    def sigmoid(self):
        '''
        Define sigmoid
        '''
        s = 1 / (1 + math.exp(-1 * self.value))
        output = Scalar(s, [self], 'sigmoid')
        output.requires_grad = self.requires_grad
        # Calculate the partial derivative ∂output/∂self = output * (1 - output)
        output.grad_wrt[self] = s * (1 - s)
        return output
    
    def __rsub__(self, other):
        '''
        Define reflected subtraction; other - self invokes this method
        '''
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        output = Scalar(other.value - self.value, [self, other], '-')
        output.requires_grad = self.requires_grad or other.requires_grad
        # Calculate the partial derivative ∂output/∂self = -1
        output.grad_wrt[self] = -1
        # Calculate the partial derivative ∂output/∂other = 1
        output.grad_wrt[other] = 1
        return output
    
    def __radd__(self, other):
        '''
        Define reflected addition; other + self invokes this method
        '''
        return self.__add__(other)
    
    def __rmul__(self, other):
        '''
        Define reflected multiplication; other * self invokes this method
        '''
        return self * other
    
    def backward(self, fn=None):
        '''
        Starting from the current node, calculate ∂self/∂node for every node in the computation graph rooted here
        Parameters
        ----
        fn : plotting function; when not None, records every backpropagation step
        Returns
        ----
        re : record of every backpropagation step
        '''
        def _topological_order():
            '''
            Return a topological ordering of the computation graph using depth-first search
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
            Propagate backward from node
            '''
            # Required for plotting but has no effect on the calculation
            node.back_prop = dict()
            # Obtain the current node's gradient in this computation graph; a node can appear in multiple computation graphs,
            # Use cg_grad to record the gradient in the current computation graph
            dnode = cg_grad[node]
            # Use node.grad to record the node's accumulated gradient
            node.grad += dnode
            for prev in node.prevs:
                # Once node's partial derivative has been calculated, propagate it backward
                # Note that gradients propagated backward to upstream nodes are accumulated
                grad_spread = dnode * node.grad_wrt[prev]
                cg_grad[prev] = cg_grad.get(prev, 0.0) + grad_spread
                node.back_prop[prev] = node.back_prop.get(prev, 0.0) + grad_spread
        
        # The current node's partial derivative is 1 because ∂self/∂self = 1; this is the starting point of backpropagation
        cg_grad = {self: 1}
        # Traverse the computation graph in reverse topological order to calculate each node's partial derivative
        ordered = reversed(_topological_order())
        re = []
        for node in ordered:
            _compute_grad_of_prevs(node)
            # Required for plotting but has no effect on the calculation
            if fn is not None:
                re.append(fn(self, 'backward'))
        return re


def _get_node_attr(node, direction='forward'):
    '''
    Node attributes
    '''
    node_type = _get_node_type(node)
    # Set the font
    res = {'fontname': 'Menlo'}
    def _forward_attr():
        if node_type == 'param':
            node_text = f'{{ grad=None | value={node.value:.2f} | {node.label}}}'
            res.update(
                dict(label=node_text, shape='record', fontsize='10', fillcolor='lightgreen', style='filled, bold'))
            return res
        elif node_type == 'computation':
            node_text = f'{{ grad=None | value={node.value:.2f} | {node.op}}}'
            res.update(
                dict(label=node_text, shape='record', fontsize='10', fillcolor='gray94', style='filled, rounded'))
            return res
        elif node_type == 'input':
            if node.label == '':
                node_text = f'input={node.value:.2f}'
            else:
                node_text = f'{node.label}={node.value:.2f}'
            res.update(dict(label=node_text, shape='oval', fontsize='10'))
            return res
    
    def _backward_attr():
        attr = _forward_attr()
        attr['label'] = attr['label'].replace('grad=None', f'grad={node.grad:.2f}')
        if not node.requires_grad:
            attr['style'] = 'dashed'
        # Improve the appearance of the plot
        # Draw a node with a dashed line if its backpropagated gradient is 0 or if it does not require gradients
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
    Determine the node type: operation, parameter, or input data
    '''
    if node.op is not None:
        return 'computation'
    if node.requires_grad:
        return 'param'
    return 'input'


def _trace(root):
    '''
    Traverse all nodes and edges in the graph
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
    Draw nodes
    '''
    node_attr = _get_node_attr(node, direction)
    uid = str(id(node)) + direction
    graph.node(name=uid, **node_attr)


def _draw_edge(graph, n1, n2, direction='forward'):
    '''
    Draw edges
    '''
    uid1 = str(id(n1)) + direction
    uid2 = str(id(n2)) + direction
    def _draw_back_edge():
        if n1.requires_grad and n2.requires_grad:
            grad = n2.back_prop.get(n1, None)
            if grad is None:
                graph.edge(uid2, uid1, arrowhead='none', color='deepskyblue')   
            elif grad == 0:
                graph.edge(uid2, uid1, style='dashed', label=f'{grad:.2f}', color='deepskyblue', fontname='Menlo')
            else:
                graph.edge(uid2, uid1, label=f'{grad:.2f}', color='deepskyblue', fontname='Menlo')
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
    Visualize the computation graph rooted at root
    Parameters
    ----
    root : Scalar, root of the computation graph
    direction : str, forward or backward propagation
    Returns
    ----
    re : Digraph, computation graph
    '''
    nodes, edges = _trace(root)
    rankdir = 'BT' if direction == 'forward' else 'TB'
    graph = Digraph(format='svg', graph_attr={'rankdir': rankdir})
    for item in nodes:
        _draw_node(graph, item, direction)
    for n1, n2 in edges:
        _draw_edge(graph, n1, n2, direction)
    return graph
