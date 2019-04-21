from reinforch.models.model import Model, DQNModel, PolicyGradientModel

models = dict(
    dqn=DQNModel,
    pg=PolicyGradientModel,
)

__all__ = ['Model', 'DQNModel', 'models', 'PolicyGradientModel']
