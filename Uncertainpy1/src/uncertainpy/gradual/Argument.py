from .Arguable import Arguable

class Argument(Arguable):
    """
    A class representing an argument in an argumentation framework.
    """

    def __init__(self, name: str, initial_weight: float, strength: float=None, attacks=None, supports=None):
        super().__init__(name=name, initial_weight=initial_weight, strength=strength, attacks=attacks, supports=supports)

    def __str__(self) -> str:
        return f"Arg({self.name}):{round(self.initial_weight, 3)}->{round(self.strength, 3)}"

    def __repr__(self) -> str:
        return self.__str__()