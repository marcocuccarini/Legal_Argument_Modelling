class SumAggregation:
    def __init__(self) -> None:
        pass

    def aggregate_strength(self, attacks, supports, state):
        aggregate = 0
        for a in attacks:
            aggregate -= state[a] * state[a.source]

        for s in supports:
            aggregate += state[s] * state[s.source]

        return aggregate
    
    def __str__(self) -> str:
        return __class__.__name__
