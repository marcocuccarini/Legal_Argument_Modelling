from .ContinuousModularModel import ContinuousModularModel
from .modular.ProductAggregation import ProductAggregation
from .modular.LinearInfluence import LinearInfluence

class ContinuousDFQuADModel(ContinuousModularModel):
    def __init__(self, BAG=None, approximator=None) -> None:
        super().__init__(BAG=BAG, approximator=approximator, aggregation=ProductAggregation(), influence=LinearInfluence())
        self.name = __class__.__name__
