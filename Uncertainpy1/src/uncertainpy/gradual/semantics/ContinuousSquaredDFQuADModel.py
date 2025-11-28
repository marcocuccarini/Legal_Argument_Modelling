from .ContinuousModularModel import ContinuousModularModel
from .modular.SquaredProductAggregation import SquaredProductAggregation
from .modular.LinearInfluence import LinearInfluence


class ContinuousSquaredDFQuADModel(ContinuousModularModel):
    def __init__(self, BAG=None, approximator=None) -> None:
        super().__init__(BAG=BAG, approximator=approximator, aggregation=SquaredProductAggregation(), influence=LinearInfluence())
        self.name = __class__.__name__
