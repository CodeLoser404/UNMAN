import numpy as np


# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
    agent_state = np.zeros(shape=[15], dtype=np.float64)
    return agent_state