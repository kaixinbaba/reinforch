from reinforch.models.model import Model, DQNModel

models = dict(
    dqn=DQNModel,
)

__all__ = ['Model', 'DQNModel', 'models']
