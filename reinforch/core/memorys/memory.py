from typing import Union

import numpy as np


class Memory(object):

    def __init__(self):
        pass

    def store(self, **kwargs):
        pass

    def sample(self, batch_size=32):
        result = self._sample(batch_size)
        self.after_sample()
        return result

    def after_sample(self):
        pass

    def _sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SimpleMatrixMemory(Memory):

    def __init__(self,
                 row_size,
                 column_size,
                 every_class_size: Union[list, tuple],
                 column_class: int = 5):
        super(SimpleMatrixMemory, self).__init__()
        self.row_size = row_size
        self.column_size = column_size
        self.memory = np.zeros([self.row_size, self.column_size])
        self.count = 0
        self.every_class_size = every_class_size
        self.column_size = column_class
        assert len(self.every_class_size) == self.column_size

    def store(self,
              state=None,
              action=None,
              reward=None,
              next_state=None,
              done=None,
              **kwargs):
        current_memory = np.hstack((state, action, reward, next_state, done))
        memory_index = self.count % self.row_size
        self.memory[memory_index, :] = current_memory
        self.count += 1

    def _sample(self, batch_size):
        batch_index = np.random.choice(self.row_size, batch_size)
        mini_batch = self.memory[batch_index, :]
        split_batch = []
        slice_from = 0
        for i in range(self.column_size):
            size = self.every_class_size[i]
            slice_to = slice_from + size
            b = mini_batch[:, slice_from:slice_to]
            split_batch.append(b)
            slice_from = slice_to
        return split_batch

    def __len__(self):
        return self.count
