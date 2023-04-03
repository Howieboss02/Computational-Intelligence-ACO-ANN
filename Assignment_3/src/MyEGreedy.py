import random
import numpy as np

class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        """
        Return an action (string) at random from possible actions
        or None if there are no possible actions (sus?)
        :param agent: of class Agent (our robot)
        :param maze: of class Maze (our maze that we go through)
        :return: (string) action or None if no action is possible
        """
        valid_actions = maze.get_valid_actions(agent)
        if len(valid_actions) == 0:
            print("We are stuck")
            return None

        return random.choice(valid_actions)

    def get_best_action(self, agent, maze, q_learning):
        """
        Return an action (string) with the highest value in q_learning
        or None if there are no possible actions (sus?)
        :param agent: of class Agent (our robot)
        :param maze: of class Maze (our maze that we go through)
        :param q_learning: of class QLearning
            (dictionary of values for each (state, action) pairs)
        :return: (string) action or None if no action is possible
        """
        valid_actions = maze.get_valid_actions(agent)
        if len(valid_actions) == 0:
            print("We are stuck")
            return None

        # Shuffle possible actions so that in case all of them are equal we pick different ones
        np.random.shuffle(valid_actions)
        state = agent.get_state(maze)
        actions_values = q_learning.get_action_values(state, valid_actions)

        return valid_actions[np.argmax(actions_values)]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        """
        Return an action (string) at random with probability epsilon
        or the best action with probability (1 - epsilon)
        or None if there are no possible actions (sus?)
        :param agent: of class Agent (our robot)
        :param maze: of class Maze (our maze that we go through)
        :param q_learning: of class QLearning
            (dictionary of values for each (state, action) pairs)
        :param epsilon: probability of getting random action
        :return: (string) action or None if no action is possible
        """
        assert 0 <= epsilon <= 1, "Epsilon should be in range [0, 1]"
        if random.random() < epsilon:
            return self.get_random_action(agent, maze)
        return self.get_best_action(agent, maze, q_learning)
