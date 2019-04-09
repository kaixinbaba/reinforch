from reinforch.utils import from_config
from reinforch.core.networks import Network


class Model(object):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def from_config(config):
        pass

class DQNModel(Model):

    def __init__(self,
                 input_size=None,
                 output_size=None,
                 last_scale=None,
                 config=None):
        super(DQNModel, self).__init__()
        self.network = Network(input_size=input_size,
                               output_size=output_size,
                               last_scale=last_scale,
                               config=config)
