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

    def fitness_function(self, distances_matrix):
        route_length = 0
        for product in range(len(self.products) - 1):
            route_length += distances_matrix[self.products[product]][self.products[product + 1]]
        self.score = route_length
        return route_length

    def create_chromosome(self, num_of_products):
        products = np.arange(num_of_products)
        np.random.shuffle(products)
        self.products = products


class Population:
    def __init__(self, mutation_prob, crossover_prob, tsp_data):
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.chromosomes = []
        self.fitness_sum = 0
        self.tsp_data = tsp_data

    def create_random_population(self, pop_size, num_of_products):
        for i in range(pop_size):
            self.chromosomes.append(Chromosome().create_chromosome(num_of_products))

    def calculate_population_fitness(self):
        fitness = 0
        for chromosome in self.chromosomes:
            fitness += chromosome.fitness_function(self.tsp_data.get_distances())
        self.fitness_sum = fitness
        return fitness




