import torch.nn as nn

from reinforch.core.networks import Layer, Input, networks, Output
from reinforch.exception import ReinforchException
from reinforch.utils import util_from_config


class Network(nn.Module):

    def __init__(self, input_size, output_size, action_dim=None, config=None, **kwargs):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.action_dim = action_dim
        self.continue_action = action_dim is not None
        self.layers = []
        self._setup(config)

    def _setup(self, config):
        # FIXME config first ?
        config = util_from_config(config)
        network = config.get('type')
        if network is None or networks.get(network) is None:
            raise ReinforchException('type {} in config is not exists!'.format(network))
        layers = config.get('layers')
        if layers is None or isinstance(layers, list):
            raise ReinforchException('Network\'s layers must be not None and list !')
        for config_layer in layers:
            layer = Layer.from_config(config_layer)
            self.layers.append(layer)

    def from_config(self, config):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def add_layers(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)


class DenseNetwork(Network):

    def __init__(self, input_size, output_size, action_dim=None, config=None):
        super(DenseNetwork, self).__init__(input_size=input_size,
                                           output_size=output_size,
                                           action_dim=action_dim,
                                           config=config)
