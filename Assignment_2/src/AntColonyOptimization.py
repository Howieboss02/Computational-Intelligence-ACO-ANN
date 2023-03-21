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
    # @param straight_factor The factor specifing how more likely the ant is to follow current direction
    def __init__(self, maze, ants_per_gen, generations, q, evaporation, straight_factor = 1, max_steps = float('inf')):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation
        self.straight_factor = straight_factor
        self.max_steps = max_steps

     # Loop that starts the shortest path process
     # @param spec Spefication of the route we wish to optimize
     # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()

        best_routes = []
        avg_routes = []

        for generation in range(1, self.generations + 1):
            routes = []
            best_route = None       
            avg_route = 0

            for ant_idx in range(self.ants_per_gen):

                #calculate route the ant take
                route = Ant(self.maze, path_specification, self.straight_factor, self.max_steps).find_route()
                if (route is not None): routes.append(route)

                # calculate best route
                if (best_route is None or route.size() < best_route.size()): best_route = route

                # use route to calculate average route if route exists
                if (route is not None): avg_route += route.size()
            
            # record best and average routes
            best_routes.append(best_route)
            avg_route /= len(routes)
            avg_routes.append(avg_route)

            # set pheromones in the maze
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(routes, self.q)

            print("generation: ", generation, ", best route: ", best_route.size(), ", avg route: ", avg_route)

        best_route = None
        for route in best_routes:
            if (best_route is None or route.size() <= best_route.size()): best_route = route

        return best_route, best_routes, avg_routes