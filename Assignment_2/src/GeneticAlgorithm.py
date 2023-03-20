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
        start_index = np.random.randint(0, high=chromosome_size)
        max_length = chromosome_size - start_index
        # Add +1 since high is exclusive
        substring_length = np.random.randint(1, high=max_length + 1)

        child_genes = np.zeros(chromosome_size)
        parent1_genes = parent1.get_genes()

        # Copy the selected substring from the parent1 to the child
        set_of_genes = set()
        for i in range(start_index, start_index + substring_length):
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

    def roulette(self, population):
        ratios = []
        chromosomes = population.get_chromosomes()
        total_sum = population.get_fitness_sum()
        for i in chromosomes:
            ratios.append(i.get_score() / total_sum)

        parents = random.choices(chromosomes, weights=ratios, k=2)
        return parents

class Chromosome:
    def __init__(self):
        self.products = []
        self.score = 0

    def to_string(self):
        return str(self.products) + " and score = " + str(self.score)

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
        self.products = products.copy()

    def size(self):
        """
        Size function for the chromosome
        :return: number of products to visit
        """
        return len(self.products)

    def fitness_function(self, tsp_data):
        route_length = 0
        distances_matrix = tsp_data.get_distances()
        # Add starting distance (start distance from the first element from the products list)
        route_length += tsp_data.get_start_distances()[self.products[0]]

        for product in range(len(self.products) - 1):
            route_length += distances_matrix[self.products[product]][self.products[product + 1]]

        # Add ending distance (end distance from the last element from the products list)
        route_length += tsp_data.get_end_distances()[self.products[len(self.products) - 1]]

        self.score = route_length
        return route_length

    def create_chromosome(self, num_of_products):
        products = np.arange(num_of_products)
        self.products = np.shuffle(products)
        products = np.arange(num_of_products)
        np.random.shuffle(products)
        self.products = products
        return self

    def get_score(self):
        return self.score

    def get_products(self):
        return self.products

    def inversion_mutation(self):
        chromosome_size = self.size()
        if chromosome_size <= 1:
            return
        position1 = np.random.randint(low=0, high=chromosome_size)
        position2 = np.random.randint(low=0, high=chromosome_size)
        while position1 == position2:
            position2 = np.random.randint(low=0, high=chromosome_size)

        gene1 = self.get_genes()[position1]
        gene2 = self.get_genes()[position2]

        self.products[position1] = gene2
        self.products[position2] = gene1

class Population:
    def __init__(self, mutation_prob, crossover_prob, tsp_data):
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.chromosomes = []
        self.fitness_sum = 0
        self.tsp_data = tsp_data

    def create_random_population(self, pop_size, num_of_products):
        chromosomes = []
        for i in range(pop_size):
            chromosomes.append(Chromosome().create_chromosome(num_of_products))
        self.chromosomes = chromosomes
        return chromosomes

    def calculate_population_fitness(self):
        fitness = 0
        for chromosome in self.chromosomes:
            fitness += chromosome.fitness_function(self.tsp_data)
        self.fitness_sum = fitness
        return fitness

    def get_chromosomes(self):
        return self.chromosomes

    def get_fitness_sum(self):
        return self.fitness_sum

