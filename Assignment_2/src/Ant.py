import random
from Route import Route
from Direction import Direction

#Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    # @param max_step The maximum amount of steps that the ant can take.
    # @param rand The random object to use for randomization.
    # @param visited The list of visited coordinates.
    def __init__(self, maze, path_specification, max_steps):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.max_steps = max_steps
        self.rand = random
        self.visited_route = []
        self.visited_map = [[False for j in range(self.maze.get_length())] for i in range(self.maze.get_width())] 

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze or None if ant didn't reach the end.
    def find_route(self):
        number_of_steps = 0
        route = Route(self.start)

        while (self.current_position != self.end and number_of_steps < self.max_steps):
            selected_direction = self.select_direction()[0]

            self.visited_map[self.current_position.get_x()][self.current_position.get_y()] = True

            # if there are no directions to be taken go back one step
            if selected_direction is None:
                self.current_position = self.visited_route.pop(-1)
                route.remove_last()
            else:
                route.add(selected_direction)

                self.visited_route.append(self.current_position)
                self.current_position = self.current_position.add_direction(selected_direction)

            number_of_steps += 1

        # print(number_of_steps)
        return route if self.current_position == self.end else None

    # Selects direction ro follow based on pheromones of surrounding fields.
    # @return Direction to follow.
    def select_direction(self):
        weights = self.maze.get_surrounding_pheromone(self.current_position).get_all_pheromones_array()
        for i in range(0, 4):
            new_position = self.current_position.add_direction(Direction(i))
            if not self.maze.in_bounds(new_position) or self.visited_map[new_position.get_x()][new_position.get_y()]:
                weights[i] = 0


        # check if there is at least one possible direction
        for weight in weights:
            if weight != 0:
                return self.rand.choices([Direction.east, Direction.north, Direction.west, Direction.south], weights)
            
        return [None]