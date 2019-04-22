from reinforch.core.memorys import PGMemory, SimpleMatrixMemory, PrioritizeMemory


def test_PGMemory():
    memory = PGMemory()
    for i in range(5):
        memory.store(1, 1, 1)
    assert len(memory) == 5
    memory.sample()
    assert len(memory) == 0


def test_SimpleMatrixMemory():
    memory = SimpleMatrixMemory(1, every_class_size=[1, 1, 1, 1, 1])
    for i in range(5):
        memory.store(1, 1, 1, 1, 1)
    assert len(memory) == 5
    batch = memory.sample(1)
    assert isinstance(batch, dict)
    assert batch.get('mini_batch') is not None


def test_PrioritizeMemory():
    memory = PrioritizeMemory(3, every_class_size=[1, 1, 1, 1, 1])
    for i in range(10):
        memory.store(1, 1, 1, 1, 1)
    assert len(memory) == 3
    batch = memory.sample(1)
    assert isinstance(batch, dict)
    assert batch.get('mini_batch') is not None
    assert batch.get('tree_index') is not None
