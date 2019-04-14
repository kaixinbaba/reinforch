import torch.nn as nn

from reinforch.core.networks import Layer
from reinforch.exception import ReinforchException
from reinforch.utils import util_from_config


class Network(nn.Module):

    def __init__(self, in_size, out_size, action_dim=None, layers: list = None, **kwargs):
        super(Network, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.action_dim = action_dim
        self.continue_action = action_dim is not None
        self.layers = []
        # self.layers = nn.ModuleList()
        self._setup(layers)

    def _setup(self, layers):
        if layers is None or not isinstance(layers, list):
            raise ReinforchException('Network\'s layers must be not None and list !')
        for config_layer in layers:
            layer = Layer.from_config(config_layer)
            self.add_layers(layer)

    @staticmethod
    def from_config(config):
        return util_from_config(config, predefine=networks)

    def forward(self, x):
        raise NotImplementedError

    def add_layers(self, layer):
        assert isinstance(layer, Layer)
        # setattr(self, layer.name, layer)
        self.layers.append(layer)


class DenseNetwork(Network):

    def __init__(self, in_size, out_size, action_dim=None, config=None, **kwargs):
        super(DenseNetwork, self).__init__(in_size=in_size,
                                           out_size=out_size,
                                           action_dim=action_dim,
                                           config=config, **kwargs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

networks = dict(
    network=Network,
    dense=DenseNetwork,
)
