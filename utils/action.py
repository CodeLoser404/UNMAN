import numpy as np


def marshal_action(action):
    action = np.array([np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)], dtype=np.float64)
    return action