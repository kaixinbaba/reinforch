from typing import Union, List, Tuple

import numpy as np


class Memory(object):

    def __init__(self):
        pass

    def store(self, **kwargs):
        """
        将数据存入记忆库.

        :param kwargs:
        :return:
        """
        pass

    def sample(self, batch_size=32):
        """
        从数据库采样batch_size个数据

        :param batch_size:
        :return:
        """

        result = self._sample(batch_size)
        self.after_sample()
        return result

    def after_sample(self):
        """
        采样结束后的钩子方法，由子类实现

        :return:
        """

        pass

    def _sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        """
        返回记忆库样本数量

        :return:
        """

        raise NotImplementedError


class SimpleMatrixMemory(Memory):
    """
    使用简单矩阵作为记忆库.

    每一行是每一次交互的训练数据
    每一列或几列是一个字段
    Example:
           state       action      reward    next_state     done
     0:[-0.80016041 -1.84146678 -0.34064761 -0.58084582 -0.29321968]
     1:[-0.17617454  0.96246334  0.82105457  0.63887006 -0.25739864]
     2:[-0.94404161 -0.46766073  0.78250654 -0.16635794  0.01809711]
     3:[ 0.77612405 -0.05981948 -0.70556892 -0.70248351 -0.18552027]
     ...
    """

    def __init__(self,
                 row_size,
                 every_class_size: Union[List[int], Tuple[int]],
                 column_class: int = 5):
        super(SimpleMatrixMemory, self).__init__()
        self.row_size = row_size
        self.column_size = sum(every_class_size)
        self.memory = np.zeros([self.row_size, self.column_size])
        self.count = 0
        self.every_class_size = every_class_size
        self.column_class = column_class
        assert len(self.every_class_size) == self.column_class

    def store(self,
              state=None,
              action=None,
              reward=None,
              next_state=None,
              done=None,
              **kwargs):
        current_memory = np.hstack((state, action, reward, next_state, done))
        assert len(current_memory) == self.column_size
        memory_index = self.count % self.row_size
        self.memory[memory_index, :] = current_memory
        self.count += 1

    def _sample(self, batch_size):
        batch_index = np.random.choice(self.row_size, batch_size)
        mini_batch = self.memory[batch_index, :]
        split_batch = []
        slice_from = 0
        for i in range(self.column_class):
            size = self.every_class_size[i]
            slice_to = slice_from + size
            b = mini_batch[:, slice_from:slice_to]
            split_batch.append(b)
            slice_from = slice_to
        return split_batch

    def __len__(self):
        return self.count
