"""
TODO
SimpleMatrixMemory              v
SumTree                         x

"""
from reinforch.core.memorys.memory import Memory, SimpleMatrixMemory

memorys = dict(
    matrix=SimpleMatrixMemory,
)


__all__ = ['SimpleMatrixMemory', 'memorys']