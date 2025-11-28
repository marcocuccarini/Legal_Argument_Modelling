class SquaredProductAggregation:
    def __init__(self) -> None:
        pass

    def aggregate_strength(self, attacks, supports, state):
        support_value = 1
        for a in attacks:
            support_value *= (1-state[a]*state[a.source])**2

        attack_value = 1
        for s in supports:
            attack_value *= (1-state[s]*state[s.source])**2

        return support_value - attack_value

    def __str__(self) -> str:
        return __class__.__name__
