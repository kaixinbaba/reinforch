"""
Collection of custom layer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforch import ReinforchException
from reinforch.utils import util_from_config


class Layer(object):

    def __init__(self, name=None):
        self.name = name

    def apply(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @staticmethod
    def from_config(config):
        config = util_from_config(config)



class Input(Layer):

    def __init__(self, aggregation_type='concat', dim=1, name=None):
        if aggregation_type == 'concat':
            # TODO use dim param
            self.layer = torch.cat
        else:
            raise ReinforchException('Unknown aggregation_type "{}"'.format(aggregation_type))
        super(Input, self).__init__(name=name)

    def apply(self, *x):
        return self.layer((*x,))


class Output(Layer):
    pass



class Linear(Layer):

    def __init__(self, in_size, out_size, name=None, bias=True):
        self.layer = nn.Linear(in_features=in_size, out_features=out_size, bias=bias)
        super(Linear, self).__init__(name=name)

    def apply(self, x):
        return self.layer(x)


class Dense(Layer):

    def __init__(self, in_size, out_size, name=None, bias=True, nonlinear='relu'):
        self.nonlinear_layer = Nonlinearity(nonlinear)
        self.linear_layer = Linear(in_size=in_size, out_size=out_size, bias=bias)
        super(Dense, self).__init__(name=name)

    def apply(self, x):
        x = self.linear_layer(x)
        return self.nonlinear_layer(x)


class PytorchLayer(Layer):
    pytorch_layers = dict(
        avg_pool1d=nn.AvgPool1d,
        avg_pool2d=nn.AvgPool2d,
        avg_pool3d=nn.AvgPool3d,
        batch_norm1d=nn.BatchNorm1d,
        batch_norm2d=nn.BatchNorm2d,
        batch_norm3d=nn.BatchNorm3d,
        conv1d=nn.Conv1d,
        conv1d_transpose=nn.ConvTranspose1d,
        conv2d=nn.Conv2d,
        conv2d_transpose=nn.ConvTranspose2d,
        conv3d=nn.Conv3d,
        conv3d_transpose=nn.ConvTranspose3d,
        linear=nn.Linear,
        dropout=nn.Dropout,
        dropout2d=nn.Dropout2d,
        dropout3d=nn.Dropout3d,
        max_pool1d=nn.MaxPool1d,
        max_pool2d=nn.MaxPool2d,
        max_pool3d=nn.MaxPool3d,
        max_unpool1d=nn.MaxUnpool1d,
        max_unpool2d=nn.MaxUnpool2d,
        max_unpool3d=nn.MaxUnpool3d,
        adaptive_avg_pool1d=nn.AdaptiveAvgPool1d,
        adaptive_avg_pool2d=nn.AdaptiveAvgPool2d,
        adaptive_avg_pool3d=nn.AdaptiveAvgPool3d,
        adaptive_max_pool1d=nn.AdaptiveMaxPool1d,
        adaptive_max_pool2d=nn.AdaptiveMaxPool2d,
        adaptive_max_pool3d=nn.AdaptiveMaxPool3d,
    )


class Nonlinearity(Layer):
    # FIXME use F or nn ?
    nonlinear_layers = dict(
        relu=F.relu,
        softmax=F.softmax,
        tanh=F.tanh,
        sigmoid=F.sigmoid,
        softmin=F.softmin,
        softplus=F.softplus,
        log_softmax=F.log_softmax,
    )

    def __init__(self, name=None, nonlinear='relu'):
        self.layer = self.nonlinear_layers.get(nonlinear)
        if self.layer is None:
            raise ReinforchException('Unknown nonlinear type "{}"'.format(nonlinear))
        super(Nonlinearity, self).__init__(name=name)

    def apply(self, x):
        return self.layer(x)


class Dropout(Layer):
    pass


class Embedding(Layer):
    pass


class Dueling(Layer):
    pass


class Conv1d(Layer):
    pass


class Conv2d(Layer):
    pass


class InternalLstm(Layer):
    pass


class Lstm(Layer):
    pass


class Flatten(Layer):
    pass


class Pool2d(Layer):
    pass
