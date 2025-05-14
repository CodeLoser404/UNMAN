import numpy as np

def normalize(value, min_value, max_value):
    return 2 * (value - min_value) / (max_value - min_value) - 1

# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
    # agent_state = np.zeros(shape=[15], dtype=np.float64)
    relative_pos = my_state[0:3] - enemy_state[0:3]
    my_attitude = my_state[3:6]
    enemy_attitude = enemy_state[3:6]
    relative_vel = my_state[6:9] - enemy_state[6:9]

    relative_pos = normalize(relative_pos, -1000, 1000)
    relative_vel = normalize(relative_vel, -100, 100)

    # 这两个已经归一化了
    my_hp = my_state[-1]
    enemy_hp = enemy_state[-1]

    return np.concatenate((relative_pos, relative_vel, my_attitude, enemy_attitude, [my_hp, enemy_hp]), axis=0) # [14]