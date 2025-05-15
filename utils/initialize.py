import numpy as np


def generate_initial_state():
    my_initial_state = np.zeros(12)
    enemy_initial_state = np.zeros(12)
    enemy_initial_state[0] = 30
    enemy_initial_state[1] = 10
    enemy_initial_state[2] = 10
    initial_state = np.append(my_initial_state, enemy_initial_state)
    return initial_state