"""
Collection of custom layer implementations.
"""
import torch.nn as nn
import torch.nn.functional as F


class Layer(object):
    pass


class Input(Layer):
    pass


class Output(Layer):
    pass


class Linear(Layer):
    pass


class Dense(Layer):
    pass


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
    nolinear_layers = dict(
        relu=F.relu,
        softmax=F.softmax,
        tanh=F.tanh,
        sigmoid=F.sigmoid,
    )


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
