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

    @staticmethod
    def OX_crossover(parent1, parent2):
        """
        Method taking two parent chromosomes
        and applying OX crossover on them to produce an offspring
        :param parent1: 1st parent chromosome
        :param parent2: 2nd parent chromosome
        :return: child chromosome
        """
        assert parent1.size() == parent2.size(), "Parent sizes should be equal"

        # Pick a random substring from the parent1
        chromosome_size = parent1.size()
        startIndex = np.random.randint(0, high=chromosome_size)
        max_length = chromosome_size - startIndex
        # Add +1 since high is exclusive
        substring_length = np.random.randint(1, high=max_length + 1)

        child_genes = np.zeros(chromosome_size)
        parent1_genes = parent1.get_genes()

        # Copy the selected substring from the parent1 to the child
        set_of_genes = set()
        for i in range(startIndex, startIndex + substring_length):
            child_genes[i] = parent1_genes[i]
            set_of_genes.add(parent1_genes[i])

        # Retrieve the genes from 2nd parent that weren't in the above substring
        parent2_contribution = []
        for gene in parent2.get_genes():
            if gene not in set_of_genes:
                parent2_contribution.append(gene)

        # Add the genes from the parent2
        idx = 0
        for i in range(chromosome_size):
            if child_genes[i] == 0:
                child_genes[i] = parent2_contribution[idx]
                idx += 1

        # Create final child chromosome
        result = Chromosome()
        result.set_genes(child_genes)
        return result


class Chromosome:
    def __init__(self):
        self.products = []
        self.score = 0

    def get_genes(self):
        """
        Getter for the order of the picked products
        :return: list of genes
        """
        return self.products

    def set_genes(self, products):
        """
        Setter for the products order in the chromosome
        :param products: new products order
        """
        self.products = products

    def size(self):
        """
        Size function for the chromosome
        :return: number of products to visit
        """
        return len(self.products)

    def fitness_function(self, tsp_data):
        route_length = 0
        for product in range(len(self.products) - 1):
            route_length += tsp_data.get_distances[self.products[product]][self.products[product + 1]]
            # add distances to route_length

    def create_chromosome(self, num_of_products):
        products = np.arange(num_of_products)
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

