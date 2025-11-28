from .ContinuousModularModel import ContinuousModularModel
from .modular.SquaredSumAggregation import SquaredSumAggregation
from .modular.QuadraticMaximumInfluence import QuadraticMaximumInfluence


class SquaredEnergyModel(ContinuousModularModel):

    def __init__(self, BAG=None, approximator=None) -> None:
        super().__init__(BAG=BAG, approximator=approximator, aggregation=SquaredSumAggregation(), influence=QuadraticMaximumInfluence())
        self.name = __class__.__name__

 