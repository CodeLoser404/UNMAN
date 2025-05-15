import numpy as np

# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
    # agent_state = np.zeros(shape=[15], dtype=np.float64)
    relative_pos = my_state[0:3] - enemy_state[0:3]
    my_attitude = my_state[3:6]
    enemy_attitude = enemy_state[3:6]
    relative_vel = my_state[6:9] - enemy_state[6:9]
    my_hp = my_state[-1]
    enemy_hp = enemy_state[-1]
    # 打开VecNormalize之后这里就不用归一化了
    return np.concatenate((relative_pos, relative_vel, my_attitude, enemy_attitude, [my_hp, enemy_hp]), axis=0) # [14]