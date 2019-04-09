import json
import os
from json import JSONDecodeError
from typing import Union

from reinforch.core.configs import Config
from reinforch.core.logger import Log, level
from reinforch.exception import ReinforchException

logging = Log(__name__, level)


def read_config(config: Union[str, dict, Config]) -> Config:
    logging.debug('config : {}, type : {}'.format(config, type(config)))
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
