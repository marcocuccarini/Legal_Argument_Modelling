from .ContinuousModularModel import ContinuousModularModel
from .modular.SumAggregation import SumAggregation
from .modular.EulerBasedInfluence import EulerBasedInfluence


class ContinuousEulerBasedModel(ContinuousModularModel):
    def __init__(self, BAG=None, approximator=None) -> None:
        super().__init__(BAG=BAG, approximator=approximator, aggregation=SumAggregation(), influence=EulerBasedInfluence())
        self.name = __class__.__name__

