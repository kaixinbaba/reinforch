from reinforch.models.model import Model, DQNModel, PolicyGradientModel, \
                                ActorModel, CriticModel

models = dict(
    dqn=DQNModel,
    pg=PolicyGradientModel,
    actor=ActorModel,
    critic=CriticModel,
)

__all__ = ['Model', 'DQNModel', 'models', 'PolicyGradientModel', 'ActorModel', 'CriticModel']
