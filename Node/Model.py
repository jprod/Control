import numpy as np


class Model:
    def __init__(self, param):
        self.param = param

    def get_output(self, error):
        out = error
        return out

    def get_params(self):
        return self.param

    def set_params(self, p):
        self.param = p
