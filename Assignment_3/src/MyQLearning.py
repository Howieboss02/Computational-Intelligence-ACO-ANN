from QLearning import QLearning
import numpy as np

class MyQLearning(QLearning):
    # This class extends QLearning
    def update_q(self, state, action, r, state_next, possible_actions, alfa=0.7, gamma=0.9):
        """
        Use bellman formula from the assignment description and updates the Q matrix using self.set_q
        :param state:       the state that we are in
        :param action:      action taken in the above state
        :param r:           reward for moving into 'state_next'
        :param state_next:  state after we apply 'action' on 'state'
        :param possible_actions:    list of possible action from the 'state_next'
        :param alfa:        learning rate
        :param gamma:       discount factor
        :return:            nothing
        """
        old_q = self.get_q(state, action)
        best_q = np.max(self.get_action_values(state_next, possible_actions))
        new_value = old_q + alfa * (r + gamma * best_q - old_q)
        self.set_q(state, action, new_value)
        return
