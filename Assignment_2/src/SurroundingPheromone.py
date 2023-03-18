from Direction import Direction

# Class containing the pheromone information around a certain point in the maze
class SurroundingPheromone:

    # Creates a surrounding pheromone object.
    # @param north the amount of pheromone in the north.
    # @param east the amount of pheromone in the east.
    # @param south the amount of pheromone in the south.
    # @param west the amount of pheromone in the west.
    def __init__(self, north, east, south, west):
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        self.total_surrounding_pheromone = east + north + south + west

    # Get surrounging pheromones in format (N/E/S/W).
    # @return array of all surrounding pheromones.
    def get_all_pheromones_array(self):
        return [self.north, self.east, self.south, self.west]

    # Get the total amount of surrouning pheromone.
    # @return total surrounding pheromone
    def get_total_surrounding_pheromone(self):
        return self.total_surrounding_pheromone

    # Get a specific pheromone level
    # @param dir Direction of pheromone
    # @return Pheromone of dir
    def get(self, dir):
        if dir == Direction.north:
            return self.north
        elif dir == Direction.east:
            return self.east
        elif dir == Direction.west:
            return self.west
        elif dir == Direction.south:
            return self.south
        else:
            return None