from Direction import Direction

# Class containing the pheromone information around a certain point in the maze
class SurroundingPheromone:

    # Creates a surrounding pheromone object.
    # @param north the amount of pheromone in the north.
    # @param east the amount of pheromone in the east.
    # @param south the amount of pheromone in the south.
    # @param west the amount of pheromone in the west.
    def __init__(self, east, north, west, south):
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        self.total_surrounding_pheromone = east + north + south + west

    # Get surrounging pheromones in format (E/N/W/S).
    # @return array of all surrounding pheromones.
    def get_all_pheromones_array(self):
        return [self.east, self.north, self.west, self.south]

    # Get the total amount of surrouning pheromone.
    # @return total surrounding pheromone
    def get_total_surrounding_pheromone(self):
        return self.total_surrounding_pheromone
    
     # Evaporate pheromone on all edges.
    # @param rho value of evaporation
    def evaporate_all(self, rho):
        self.north *= (1 - rho)
        self.east *= (1 - rho)
        self.south *= (1 - rho)
        self.west *= (1 - rho)
    
    # Add a pheromone value to the direction edge.
    # @param Direction in which to add pheromone
    # @param value of the pheromone to add
    def add_to_direction(self, dir, value):
        if dir == Direction.north:
            self.north += value
        elif dir == Direction.east:
            self.east += value
        elif dir == Direction.west:
            self.west += value
        elif dir == Direction.south:
            self.south += value

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