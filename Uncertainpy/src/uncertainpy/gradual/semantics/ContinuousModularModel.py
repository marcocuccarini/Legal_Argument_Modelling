from .Model import Model


class ContinuousModularModel(Model):
    def __init__(self, aggregation=None, influence=None, BAG=None, approximator=None) -> None:
        super().__init__(BAG=BAG, approximator=approximator, aggregation=aggregation, influence=influence)
        self.name = __class__.__name__

    def compute_derivative_at(self, state):
        derivatives = {}
        for a in self.arguable_strength:
            aggregate_strength = self.aggregation.aggregate_strength(a.attacks, a.supports, state)
            derivative = self.influence.compute_strength(a.initial_weight, aggregate_strength)
            derivative -= state[a]

            derivatives[a] = derivative

        return derivatives

    def __repr__(self) -> str:
        return super().__repr__(__name__)

    def __str__(self) -> str:
        return super().__str__(__name__)
