class Memory(object):

    def __init__(self):
        pass

    def store(self, *args):
        pass

    def sample(self, batch_size=32):
        self._sample(batch_size)
        self.after_sample()

    def after_sample(self):
        pass

    def _sample(self, batch_size):
        pass
