"""
Reinforch 工具类

"""
import json
import os
from json import JSONDecodeError
from typing import Union

import torch
import numpy as np

from reinforch.core.configs import Config
from reinforch.core.logger import Log
from reinforch.exception import ReinforchException

logging = Log(__name__)

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


def read_config(config: Union[str, dict, Config]) -> Config:
    _check_config_type(config)
    result = None
    if isinstance(config, str):
        if os.path.exists(config) and os.path.isfile(config):
            # read file
            with open(config, 'r', encoding='utf-8') as f:
                json_str = f.read()
        else:
            # json string
            json_str = config
        try:
            data = json.loads(json_str)
            result = Config(data)
        except JSONDecodeError as e:
            logging.error(e)
    elif isinstance(config, dict):
        result = Config(config)
    elif isinstance(config, Config):
        result = config
    if result is None:
        raise ReinforchException(
            "Can't read config, please check! The argument config is [{}], type is [{}]".format(config, type(config)))
    return result


def _check_config_type(config):
    if not (isinstance(config, str) or isinstance(config, dict) or isinstance(config, Config)):
        raise ReinforchException('config type is not correct! type [{}]'.format(type(config)))


def from_config(config: Union[str, dict, Config], predefine: dict = None, default_object=None, kwargs: dict = None):
    config = read_config(config)
    try:
        target_type = config.pop('type')
    except Exception as e:
        raise ReinforchException('Config must contain "type" key to init the object')
    if predefine is None:
        if default_object is None or not callable(default_object):
            raise ReinforchException('Please provide a default callable object')
        target = default_object
    else:
        target = predefine.get(target_type)
        if target is None:
            raise ReinforchException('Can\'t find match type [{}]'.format(target_type))
    kw = dict(config)
    if kwargs is not None:
        kw.update(kwargs)
    return target(**kw)


def obj2tensor(obj, feed_network: bool = True, target_type=FloatTensor, target_shape: Union[list, tuple] = None):
    """
    :param obj:
    :param feed_network: if True, reshape tensor to matrix
    :param target_type: Cast to which type tensor
    :param target_shape
    :return:
    """
    if isinstance(obj, Tensor):
        return tensor2tensor(obj, feed_network=feed_network, target_type=target_type, target_shape=target_shape)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return tensor2tensor(Tensor(obj), feed_network=feed_network, target_type=target_type, target_shape=target_shape)
    else:
        # TODO other type
        try:
            return Tensor(obj)
        except TypeError:
            raise ReinforchException('The obj type {} can not be casted to Tensor'.format(type(obj)))


def tensor2tensor(tensor, feed_network: bool = True, target_type=FloatTensor, target_shape: Union[list, tuple] = None):
    if tensor is None:
        raise ReinforchException("'None' type can not be casted to Tensor")
    # cast type
    tensor = tensor.type(target_type)
    if target_shape is not None:
        return tensor.view(target_shape)
    elif feed_network:
        size = len(tensor.size())
        if size == 1:
            return tensor.view(1, -1)
        elif size == 2:
            return tensor
        else:
            # FIXME maybe needn't raise
            raise ReinforchException('tensor\'s size more than 2, please specified [target_shape]')
    return tensor


util_from_config = from_config
t2t = tensor2tensor
o2t = obj2tensor
