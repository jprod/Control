import numpy as np


class Effector:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = []
        self.integral = []

    def get_output(self, error):
        if len(self.integral) != len(error):
            self.integral = np.zeros(len(error))
        if len(self.prev_error) != len(error):
            self.prev_error = np.zeros(len(error))
        out = np.array([])
        for e in range(len(error)):
            derivative = error[e] - self.prev_error[e]
            self.prev_error[e] = error[e]
            self.integral[e] = self.integral[e] + error[e]
            out = np.append(out, self.Kp[e]*error[e] + self.Ki[e] *
                       self.integral[e] + self.Kd[e]*derivative)
        return out

    def get_params(self):
        return self.Kp, self.Ki, self.Kd

    def set_params(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
