import math
import random

import numpy as np
import scipy


def chaotic_controller(behavioral_model, threshold, error):
    # avg absolute error
    avg_abs_error = abs(np.average(error))
    # if there is a significant error, adjust according the magnitude of the error
    if avg_abs_error > threshold:
        weights = np.random.choice([1, -1], size=error.shape)
        control_signal = behavioral_model @ (weights * error)
    else:
        control_signal = np.zeros(error.shape[0])
    return control_signal

def prop_controller(behavioral_model, error):
    control_signal = behavioral_model @ error
    return control_signal




def chaotic_update(prob, threshold, error):
    avg_abs_error = abs(np.average(error))
    if avg_abs_error > threshold:
        magnitude = np.random.choice([-1, 1]) * avg_abs_error
        r = round(np.random.uniform(0, magnitude), 3)
        p = [np.asarray([
            abs(i - r) if abs(i - r) <= 1 else i for i in prob])][0]
        return p
    return prob


def entropy(prob):
    '''
    Calculate the entropy of a probability vector
    '''
    e = [-((p * math.log(p, 2)) + ((1-p) * math.log(1-p,  2)))
         for p in prob]
    attention = np.diag(e)
    return attention


def matrix_sigmoid(x):
    '''
    Helper sigmoid function where the intercept is 0.5
    '''
    return 1 / (1 + np.exp(-x))

def zero_sigmoid(x):
    '''
    Helper sigmoid function where the intercept is 0
    '''
    return (1 / (1 + np.exp(-x))) - 0.5

def adjustable_sigmoid(x, slope):
    '''
    Helper sigmoid function where the intercept is 0 and the slope is adjustable
    '''
    return 2*((1 / (1 + np.exp(-x*slope))) - 0.5)


def abs_zero_sigmoid(x):
    '''
    Helper sigmoid function where the intercept is 0
    '''
    return 2*abs((1 / (1 + np.exp(-x))) - 0.5)

def dynamic_sigmoid(i, x):
    '''
    Helper sigmoid function where the intercept is a value of i (list)
    '''
    y = np.exp(np.clip(-x, -100, 100))  # avoid runover into infinity.
    out = np.asarray([1 /
                      (1 + (
                          (1 - np.clip(i[j], 1e-50, 1-1e-50)) / np.clip(i[j], 1e-50, 1-1e-50)) * y[j])

                      for j in range(len(i))])  # changed clips to keep sigmoid function from getting stuck at 0 or 1
    return out
