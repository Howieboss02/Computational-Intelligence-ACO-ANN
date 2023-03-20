import time
from Maze import Maze
from PathSpecification import PathSpecification
from Ant import Ant

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation, max_steps = float('inf')):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation
        self.max_steps = max_steps

     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()

        for n, generation in enumerate(range(self.generations)):
            routes = []
            best_route = None
            avg_route = 0

            for ant_idx in range(self.ants_per_gen):
                route = Ant(self.maze, path_specification, self.max_steps).find_route()
                if (route is not None): routes.append(route)

                if (best_route is None): best_route = route
                elif (route is not None and route.size() < best_route.size()): best_route = route
                if (route is not None): avg_route += route.size()
            
            avg_route /= len(routes)

            self.maze.evaporate(self.evaporation)
            for route in routes:
                self.maze.add_pheromone_route(route, self.q)

            print("generation: ", n, ", best route: ", best_route.size(), ", avg route: ", avg_route)
        best_route = None
        for ant_idx in range(self.ants_per_gen):
            route = Ant(self.maze, path_specification, self.max_steps).find_route()
            if (best_route is None): best_route = route
            elif (route is not None and route.size() < best_route.size()): best_route = route

        return best_route