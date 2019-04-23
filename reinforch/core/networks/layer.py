"""
各种神经网络层的对象定义.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforch import ReinforchException
from reinforch.utils import util_from_config


class Layer(nn.Module):
    """
    所有自定义网络层的父类对象，继承自nn.Module

    """

    def __init__(self, name=None):
        super(Layer, self).__init__()
        self.name = name

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def from_config(config):
        return util_from_config(config, predefine=layers)


class Input(Layer):

    def __init__(self, aggregation_type='concat', dim=1, name=None):
        if aggregation_type == 'concat':
            # TODO use dim param
            self.layer = torch.cat
        else:
            raise ReinforchException('Unknown aggregation_type "{}"'.format(aggregation_type))
        super(Input, self).__init__(name=name)

    def forward(self, *x):
        return self.layer((*x,))


class Output(Layer):
    pass


class Linear(Layer):

    def __init__(self, in_size, out_size, name=None, bias=True):
        super(Linear, self).__init__(name=name)
        self.layer = nn.Linear(in_features=in_size, out_features=out_size, bias=bias)
        self.layer.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.layer(x)


class Dense(Layer):

    def __init__(self, in_size, out_size, name=None, bias=True, nonlinear='relu'):
        super(Dense, self).__init__(name=name)
        self.nonlinear_layer = Nonlinearity(nonlinear=nonlinear)
        self.linear_layer = Linear(in_size=in_size, out_size=out_size, bias=bias)

    def forward(self, x):
        x = self.linear_layer(x)
        return self.nonlinear_layer(x)


class Dueling(Layer):

    def __init__(self, in_size, out_size, name=None, bias=True, nonlinear='relu'):
        super(Dueling, self).__init__(name=name)
        self.A = Linear(in_size=in_size, out_size=1, bias=bias)
        self.V = Linear(in_size=in_size, out_size=out_size, bias=bias)

    def forward(self, x):
        A = self.A(x)
        V = self.V(x)
        Q = V + (A - torch.mean(A))
        return Q


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
        super(Nonlinearity, self).__init__(name=name)
        self.layer = self.nonlinear_layers.get(nonlinear)
        if self.layer is None:
            raise ReinforchException('Unknown nonlinear type "{}"'.format(nonlinear))

    def forward(self, x):
        x = self.layer(x)
        return x


class Dropout(Layer):
    pass


class Embedding(Layer):
    pass


class Conv1d(Layer):
    pass


class Conv2d(Layer):

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 name=None):
        super(Conv2d, self).__init__(name=name)
        self.layer = nn.Conv2d(in_channels=in_size,
                               out_channels=out_size,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias)

    def forward(self, x):
        return self.layer(x)


class MaxPool2d(Layer):

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False,
                 name=None):
        super(MaxPool2d, self).__init__(name=name)
        self.layer = nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  return_indices=return_indices,
                                  ceil_mode=ceil_mode)

    def forward(self, x):
        return self.layer(x)


class InternalLstm(Layer):
    pass


class Lstm(Layer):
    pass


class Flatten(Layer):

    def __init__(self, name=None):
        super(Flatten, self).__init__(name=name)

    def forward(self, x):
        return x.view(-1)


class Pool2d(Layer):
    pass


layers = dict(
    input=Input,
    output=Output,
    tf_layer=PytorchLayer,
    nonlinearity=Nonlinearity,
    dropout=Dropout,
    flatten=Flatten,
    pool2d=Pool2d,
    embedding=Embedding,
    linear=Linear,
    dense=Dense,
    dueling=Dueling,
    conv1d=Conv1d,
    conv2d=Conv2d,
    internal_lstm=InternalLstm,
    lstm=Lstm
)
