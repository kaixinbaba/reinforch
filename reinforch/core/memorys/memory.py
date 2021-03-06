from typing import Union, List, Tuple

import numpy as np


class Memory(object):

    def __init__(self,
                 every_class_size: Union[List[int], Tuple[int]] = None,
                 column_class: int = 5):
        self.column_size = sum(every_class_size) if every_class_size else 0
        self.every_class_size = every_class_size
        self.column_class = column_class
        assert len(self.every_class_size) == self.column_class

    def store(self, **kwargs):
        """
        将数据存入记忆库.

        :param kwargs:
        :return:
        """

        raise NotImplementedError

    def sample(self, batch_size=None) -> dict:
        """
        从数据库采样batch_size个数据

        :param batch_size:
        :return: dict
        """

        raise NotImplementedError

    def __len__(self):
        """
        返回记忆库样本数量

        :return:
        """

        raise NotImplementedError

    def _column_split(self, mini_batch):
        split_batch = []
        slice_from = 0
        for i in range(self.column_class):
            size = self.every_class_size[i]
            slice_to = slice_from + size
            b = mini_batch[:, slice_from:slice_to]
            split_batch.append(b)
            slice_from = slice_to
        return split_batch


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
                 every_class_size: Union[List[int], Tuple[int]] = None,
                 column_class: int = 5):
        super(SimpleMatrixMemory, self).__init__(every_class_size=every_class_size,
                                                 column_class=column_class)
        self.row_size = row_size
        self.memory = np.zeros([self.row_size, self.column_size])
        self.count = 0

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

    def sample(self, batch_size=None):
        batch_index = np.random.choice(self.row_size, batch_size)
        mini_batch = self.memory[batch_index, :]
        return {'mini_batch': self._column_split(mini_batch)}

    def __len__(self):
        return self.count


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def __len__(self):
        return int(self.tree[0])


class PrioritizeMemory(Memory):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # epsilon = 0.01  # small amount to avoid zero priority
    # alpha = 0.6  # [0~1] convert the importance of TD error to priority
    # beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity,
                 every_class_size: Union[List[int], Tuple[int]] = None,
                 e=0.01,
                 a=0.6,
                 b=0.4,
                 column_class: int = 5):
        super(PrioritizeMemory, self).__init__(every_class_size=every_class_size,
                                               column_class=column_class)
        self.tree = SumTree(capacity)
        self.epsilon = e
        self.alpha = a
        self.beta = b
        self.column_class = column_class

    def store(self,
              state=None,
              action=None,
              reward=None,
              next_state=None,
              done=None,
              **kwargs):
        transition = np.hstack((state, [action, reward], next_state, done))
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, batch_size=None):
        b_idx, b_memory, _ = np.empty((batch_size,), dtype=np.int32), \
                             np.empty((batch_size, self.tree.data[0].size)), \
                             np.empty(
                                 (batch_size, 1))
        pri_seg = len(self.tree) / batch_size  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            b_idx[i], b_memory[i, :] = idx, data

        return {'tree_index': b_idx, 'mini_batch': self._column_split(b_memory)}

    def update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return len(self.tree)


class PGMemory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self,
              state=None,
              action=None,
              reward=None,
              **kwargs):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def sample(self, batch_size=None) -> dict:
        mini_batch = [np.vstack(self.states), np.vstack(self.actions), np.vstack(self.rewards)]
        result = {'mini_batch': mini_batch}
        self.states.clear()
        self.rewards.clear()
        self.actions.clear()
        return result

    def __len__(self):
        return len(self.states)
