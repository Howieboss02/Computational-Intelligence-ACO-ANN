from QLearning import QLearning
import numpy as np

class MyQLearning(QLearning):
    # This class extends QLearning
    def update_q(self, state, action, r, state_next, possible_actions, alfa=0.7, gamma=0.9):
        """
        Use bellman formula from the assignment and update the q using self.set_q
        :param state:
        :param action:
        :param r:
        :param state_next:
        :param possible_actions:
        :param alfa:
        :param gamma:
        :return:
        """
        old_q = self.get_q(state, action)
        best_q = np.max(self.get_action_values(state_next, possible_actions))
        new_value = old_q + alfa * (r + gamma * best_q - old_q)
        self.set_q(state, action, new_value)
        return
