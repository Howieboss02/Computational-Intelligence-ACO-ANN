import traceback
import sys
import copy
from Direction import Direction
from SurroundingPheromone import SurroundingPheromone
from Coordinate import Coordinate

# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.initialize_pheromones()

    # Initialize pheromones to a start value.
    def initialize_pheromones(self):
        self.pheromones = []
        for i in range(self.width):
            self.pheromones.append([])
            for j in range(self.length):
                north = 0
                south = 0
                east = 0
                west = 0
                if i - 1 >= 0 and self.walls[i - 1][j] != 0:
                    west = 1
                if j + 1 < self.length and self.walls[i][j + 1] != 0:
                    south = 1
                if i + 1 < self.width and self.walls[i + 1][j] != 0:
                    east = 1
                if j - 1 >= 0 and self.walls[i][j - 1] != 0:
                    north = 1

                # append to list
                self.pheromones[i].append(SurroundingPheromone(east, north, west, south))

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # add pheromone to a certain edge
    # @param position Coordinate of the start of the edge
    # @param pheromone amount of pheromone to add
    # @param direction Direction of the edge
    def add_pheromone(self, position, pheromone, direction):
        if self.in_bounds(position):
            self.pheromones[position.get_x()][position.get_y()].add_to_direction(direction, pheromone)

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        L = route.size()
        position = route.start

        for direction in route.get_route():
            position = position.add_direction(direction)
            self.add_pheromone(position, q / L, direction)

     # Update pheromones for a list of routes
     # @param routes A list of routes
     # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q):
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    
    #TODO fix this

    def evaporate(self, rho):
       
       for i in range(self.get_width()):
           for j in range(self.get_length()):
            self.pheromones[i][j].evaporate_all(rho)

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns a the amount of pheromones on the neighbouring positions (E/N/W/S).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position):
        if self.in_bounds(position):
            return self.pheromones[position.get_x()][position.get_y()]
        return None

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length)

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])
            
            #make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])
            
            for y in range(length):
                line = lines[y+1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()