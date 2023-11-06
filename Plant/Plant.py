import numpy as np


class Plant:

    def __init__(self, transition_matrix, transition_function, state):
        self.transition_matrix = transition_matrix
        self.transition_function = transition_function
        self.state = state

    def transition(self, control_signal):
        self.state = self.transition_function(
            control_signal=control_signal, transition_matrix=self.transition_matrix, state=self.state)
        return self.state
