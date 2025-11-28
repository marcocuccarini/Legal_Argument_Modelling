class Model:
    def __init__(self, BAG=None, approximator=None, aggregation=None, influence=None) -> None:
        self.BAG = BAG
        self.approximator = approximator
        self.aggregation = aggregation
        self.influence = influence
        self.arguable_strength = {}
        self.name = __class__.__name__

    def solve(self, delta, epsilon, verbose=True, generate_plot=False):
        if type(verbose) != bool:
            raise TypeError("verbose must be a boolean")

        if type(generate_plot) != bool:
            raise TypeError("generate_plot must be a boolean")

        if (type(delta) != float and type(delta) != int):
            raise TypeError("delta must be a float or integer")

        if (type(epsilon) != float and type(epsilon) != int):
            raise TypeError("epsilon must be a float or integer")

        if self.approximator is None:
            raise AttributeError("Model does not have approximator attached")

        if self.BAG is None:
            raise AttributeError("Model does not have BAG attached")

        result = self.approximator.approximate_solution(delta, epsilon, verbose, generate_plot)
        return result

    def __repr__(self, name) -> str:
        return f"{name}({self.BAG}, {self.approximator}, {self.arguable_strength})"

    def __str__(self, name) -> str:
        return f"{name} - BAG: {self.BAG}, Approximator: {self.approximator}, Argument strength: {self.arguable_strength})"
