"""
TODO
SimpleMatrixMemory              v
SumTree                         x

"""
from reinforch.core.memorys.memory import Memory, SimpleMatrixMemory, PrioritizeMemory,\
    PGMemory

memorys = dict(
    matrix=SimpleMatrixMemory,
    prioritize=PrioritizeMemory,
    pg=PGMemory,
)

__all__ = ['PrioritizeMemory', 'SimpleMatrixMemory', 'memorys', 'Memory',
           'PGMemory']
