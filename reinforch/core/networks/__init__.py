from reinforch.core.networks.layer import Layer, Input, Output, PytorchLayer, Nonlinearity, Dropout, Flatten, Pool2d, \
    Embedding, Linear, Dense, Dueling, Conv1d, Conv2d, InternalLstm, Lstm

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
]
