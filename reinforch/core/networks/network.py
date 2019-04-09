from reinforch.core.networks import Layer, Input, Output

import torch.nn as nn
import torch


class Network(nn.Module):

    def __init__(self, in_size, out_size, config=None):
        super(Network, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.layers = []
        self._setup(config)

    def _setup(self, config):
        input_ = Input()
        self.layers.append(input_)


    def forward(self, x):
        raise NotImplementedError

    def add_layers(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)

