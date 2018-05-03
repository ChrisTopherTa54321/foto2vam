# Run the genetic algorithm
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy

class Algorithm:

    def __init__(self, params):
        self.toolbox = base.Toolbox()
        self.params = params

        # We want to find the minimum value
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # Each 'individual' will be a list
        creator.create("Individual", list, fitness=creator.FitnessMin)


        # Attribute generator
        self.toolbox.register("attr_float", params.initIndividual)
        # Structure initializers
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_float)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", params.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", params.mutate,  toolbox=self.toolbox, mutProb=0.15)
        self.toolbox.decorate("mate", params.checkBounds())
        self.toolbox.decorate("mutate", params.checkBounds())
        self.toolbox.register("map", params.map)
        #self.toolbox.decorate("mate", params.registerChange())
        #self.toolbox.decorate("mutate", params.registerChange())
        self.toolbox.register("select", tools.selTournament, tournsize=3)


    def run(self, pop=100):
        print("Run algorithm with {} individuals".format(pop))
        pop = self.toolbox.population(pop)


        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        stats.register("saveBest", self.params.saveBest, hof)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=4000,
                                       stats=stats, halloffame=hof, verbose=True)