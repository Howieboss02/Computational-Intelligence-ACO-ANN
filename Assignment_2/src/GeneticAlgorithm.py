import random
from TSPData import TSPData
import numpy as np

# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        return []


class Chromosome:
    def __init__(self):
        self.products = []
        self.score = 0

    def fitness_function(self, tsp_data):
        route_length = 0
        for product in range(len(self.products) - 1):
            route_length += tsp_data.get_distances[self.products[product]][self.products[product + 1]]
            # add distances to route_length

    def create_chromosome(self, num_of_products):
        products = np.arrange(num_of_products)
        self.products = np.shuffle(products)


class Population:
    def __init__(self, mutation_prob, crossover_prob):
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.chromosomes = []
        self.fitness_sum = 0

    def create_population(self, pop_size, num_of_products):
        for i in range(pop_size):
            self.chromosomes.append(Chromosome().create_chromosome(num_of_products))




