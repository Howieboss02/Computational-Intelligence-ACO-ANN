import random
from Route import Route
from Direction import Direction

#Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    # @param max_step The maximum amount of steps that the ant can take.
    def __init__(self, maze, path_specification, max_steps):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.max_steps = max_steps
        self.rand = random

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze or None if ant didn't reach the end.
    def find_route(self):
        number_of_steps = 0
        route = Route(self.start)

        while (self.current_position != self.end and number_of_steps < self.max_steps):
            selected_direction = self.select_direction()[0]

            # if there are no directions to be taken return empty route
            if selected_direction is None:
                return None

            route.add(selected_direction)
            self.current_position = self.current_position.add_direction(selected_direction)

        return route if self.current_position == self.end else None

    # Selects direction ro follow based on pheromones of surrounding fields.
    # @return Direction to follow.
    def select_direction(self):
        weights = self.maze.get_surrounding_pheromone(self.current_position).get_all_pheromones_array()

        # check if there is at least one possible direction
        for weight in weights:
            if weight != 0:
                return self.rand.choices([Direction.north, Direction.east, Direction.south, Direction.west], weights)
            
        return None