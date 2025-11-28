from ..Argument import Argument

class Approximator:
    

    def __init__(self, ads, time=0, name="") -> None:
        self.ads = ads
        self.time = time
        self.name = name
        self.graph_data = {} #data for plotting the graph of the strength evolution?

    def initialise_approximation(self):

        arguable_strength = {}
        for a in self.ads.BAG.arguable:
            arguable_strength[a] = a.initial_weight

        self.ads.arguable_strength = arguable_strength


    def rewrite_arrays(self):
        for a in self.ads.arguable_strength:
            a.strength = self.ads.arguable_strength[a]

    def perform_iteration(delta, epsilon):
        return None

    def initialise_graph_data(self):
        #plot only arguments and edges that are attacked or supported
        for a in self.ads.arguable_strength:
            if type(a)==Argument or a.attacks or a.supports: 
                self.graph_data[a] = [(0, a.initial_weight)]

    def update_graph_data(self, time):
        for a in self.graph_data:
            self.graph_data[a].append((time, self.ads.arguable_strength[a]))

    def approximate_solution(self, delta, epsilon, verbose=False, generate_plot=False):
        self.initialise_approximation()

        if generate_plot:
            self.initialise_graph_data()

        time = 0
        time_limit = 99999999999
        max_derivative = 0

        while True:
            max_derivative = self.perform_iteration(delta, epsilon)
            time += delta

            if generate_plot:
                self.update_graph_data(time)

            if(max_derivative < epsilon or time >= time_limit):
                break

        self.rewrite_arrays()

        if (verbose):
            print_args = '\n'.join([str(x) for x in self.ads.arguable_strength if x.attacks or x.supports or type(x)==Argument])
            print(f"{self.ads.name}, {self.ads.approximator.name}\nTime: {time}\n{print_args}\n")

        return max_derivative
