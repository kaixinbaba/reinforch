"""
神经网络模块，封装了各种网络结构及网络层对象，可以通过配置初始化.

"""
from reinforch.core.networks.layer import Layer, Input, Output, PytorchLayer, Nonlinearity, Dropout, Flatten, Pool2d, \
    Embedding, Linear, Dense, Dueling, Conv1d, Conv2d, InternalLstm, Lstm, layers

from reinforch.core.networks.network import Network, DenseNetwork, networks

__all__ = [
    'layers',
    'Layer',
    'Input',
    'Output',
    'PytorchLayer',
    'Nonlinearity',
    'Dropout',
    'Flatten',
    'Pool2d',
    'Embedding',
    'Linear',
    'Dense',
    'Dueling',
    'Conv1d',
    'Conv2d',
    'InternalLstm',
    'Lstm',
    'Network',
    'networks',
]
