import random
from TSPData import TSPData
import numpy as np


# TSP problem solver using genetic algorithms.


class GeneticAlgorithm:
    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size, mutation_rate, crossover_probability):
        self.generations = generations
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_probability = crossover_probability

        self.population_fitness = []
        self.best_route_across_generation = []
        self.population = None

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        population = Population(mutation_prob=self.mutation_rate,
                                crossover_prob=self.crossover_probability,
                                tsp_data=tsp_data)

        num_of_products = len(tsp_data.get_start_distances())
        population.create_random_population(self.pop_size, num_of_products)
        population.calculate_population_fitness()
        self.population_fitness.append(population.get_fitness_sum())
        self.best_route_across_generation.append(population.take_best_chromosome().get_score())

        for i in range(self.generations):
            print("Generation ", i + 1)
            chromosomes = []
            for j in range(self.pop_size):
                # if population.get_fitness_sum() < 6000000:
                parents = population.roulette_random()
                # else:
                # parents = population.roulette_two_best()
                # if random.random() <= 0.5:
                #     parents = population.roulette_sort_and_take_from_n((int)(0.2 * self.pop_size))
                # else:
                #     parents = population.roulette_random()


                # Crossover
                rand = random.random()
                if rand < self.crossover_probability:
                    child = GeneticAlgorithm.OX_crossover(parents[0], parents[1])
                else:
                    child = parents[0]

                # Mutation
                rand = random.random()
                if rand < self.mutation_rate:
                    child.inversion_mutation()

                chromosomes.append(child)

            population.create_successor_population(chromosomes)
            self.population_fitness.append(population.get_fitness_sum())
            best_route = population.take_best_chromosome().get_score()
            print("Length of the best route = ", 1.0 / best_route)
            self.best_route_across_generation.append(best_route)

        self.population = population
        result = np.array(population.take_best_chromosome().get_genes()) - 1
        return result.astype(int)

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
        child_genes = child_genes.astype(int)
        result = Chromosome()
        result.set_genes(child_genes)
        return result


class Chromosome:
    def __init__(self):
        self.products = []
        self.score = 0

    def to_string(self):
        return str(self.products) + " and score = " + str((1.0 / self.score))

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
        route_length += tsp_data.get_start_distances()[self.products[0] - 1]

        for product in range(len(self.products) - 1):
            route_length += distances_matrix[self.products[product] - 1][self.products[product + 1] - 1]

        # Add ending distance (end distance from the last element from the products list)
        route_length += tsp_data.get_end_distances()[self.products[len(self.products) - 1] - 1]

        # self.score = route_length
        self.score = (float)(1.0 / route_length)
        # return route_length
        return (float)(1.0 / route_length)

    def create_chromosome(self, num_of_products):
        products = np.arange(num_of_products)
        products += 1
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

    def create_successor_population(self, chromosomes):
        self.chromosomes = chromosomes
        self.fitness_sum = self.calculate_population_fitness()

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

    # def take_best_chromosome(self):
    #     curr_best_score = 2 ** 62
    #     best_chromosome = Chromosome()
    #     for chrom in self.chromosomes:
    #         score = chrom.get_score()
    #         if score < curr_best_score:
    #             best_chromosome = chrom
    #             curr_best_score = score
    #     return best_chromosome
    def take_best_chromosome(self):
        curr_best_score = -1
        best_chromosome = Chromosome()
        for chrom in self.chromosomes:
            score = chrom.get_score()
            if score > curr_best_score:
                best_chromosome = chrom
                curr_best_score = score
        return best_chromosome

    def roulette_random(self):
        ratios = []
        chromosomes = self.get_chromosomes()
        total_sum = self.get_fitness_sum()
        for i in chromosomes:
            ratios.append(i.get_score() / total_sum)

        # print("Ratios before -1:", ratios)
        # ratios = 1 - np.array(ratios)
        # print(ratios)
        parents = random.choices(chromosomes, weights=ratios, k=2)
        return parents

    # Needs a change of direction
    def roulette_two_best(self):
        m1 = m2 = float('inf')
        c1 = c2 = None
        for chrom in self.chromosomes:
            if chrom.get_score() <= m1:
                m1, m2 = chrom.get_score(), m1
                c1, c2 = chrom, c1
            elif chrom.get_score() < m2:
                m2 = chrom.get_score()
                c2 = chrom
        return [c1, c2]

    # Needs a change of direction
    def roulette_sort_and_take_from_n(self, n):
        self.chromosomes = sorted(self.chromosomes, key=lambda x: x.get_score(), reverse=False)
        if random.random() < 0.8:
            return random.choices(self.chromosomes[:n], k=2)
        else:
            return random.choices(self.chromosomes, k=2)
