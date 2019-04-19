"""
TODO
SimpleMatrixMemory              v
SumTree                         x

"""
from reinforch.core.memorys.memory import Memory, SimpleMatrixMemory, PrioritizeMemory

memorys = dict(
    matrix=SimpleMatrixMemory,
    prioritize=PrioritizeMemory,
)

__all__ = ['PrioritizeMemory', 'SimpleMatrixMemory', 'memorys', 'Memory']
