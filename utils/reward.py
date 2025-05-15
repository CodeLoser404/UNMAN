import numpy as np

hit_step_count = 0
miss_step_count = 0
is_hit = False
has_hit = False

def reset_count():
    global hit_step_count, miss_step_count, has_hit, is_hit
    hit_step_count = 0
    miss_step_count = 0
    has_hit = False
    is_hit = False

def calculate_angle_reward(my_pos, enemy_pos, my_attitude):
    # 计算指向敌机的向量
    target_direction = enemy_pos - my_pos
    target_direction = target_direction / np.linalg.norm(target_direction)  # 归一化

    # 提取 pitch（俯仰）和 yaw（偏航）
    _, pitch, yaw = my_attitude  # 假设为弧度制。如果是角度制，请加 np.deg2rad()

    # 根据 pitch 和 yaw 计算飞机的朝向向量（前向方向）
    forward_vector = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch)
    ])
    forward_vector = forward_vector / np.linalg.norm(forward_vector)  # 单位化

    # 计算夹角余弦值（越接近1越对准）
    cos_theta = np.dot(forward_vector, target_direction)
    angle_reward = 10 * cos_theta if cos_theta > 0.95 else cos_theta
    return angle_reward


# This is the reward calculation function. We provide current state and previous state for you.
def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    # 距离奖励(越近越好)
    my_pos = my_state[0:3]
    enemy_pos = enemy_state[0:3]
    prev_my_pos = prev_my_state[0:3]
    prev_enemy_pos = prev_enemy_state[0:3]

    prev_dist = np.linalg.norm(prev_my_pos - prev_enemy_pos)
    cur_dist = np.linalg.norm(my_pos - enemy_pos)
    distance_reward = (prev_dist - cur_dist) * 10
    if cur_dist > 500:  # 太远了
        distance_reward = -500
    # elif cur_dist < 80:  # 太近了
    #     distance_reward = -500

    # 角度奖励(指向敌机)
    my_attitude = my_state[3:6]
    angle_reward = calculate_angle_reward(my_pos, enemy_pos, my_attitude)

    # 血量奖励(奖励让敌机受伤,避免自己受伤)
    my_hp = my_state[-1]
    enemy_hp = enemy_state[-1]
    prev_my_hp = prev_my_state[-1]
    prev_enemy_hp = prev_enemy_state[-1]
    hp_reward = 0
    global is_hit, has_hit, hit_step_count, miss_step_count
    if my_hp < prev_my_hp:  # 我方受伤
        hp_reward -= 500

    if prev_enemy_hp - enemy_hp == 0:
        if is_hit:  # 之前击中了,现在没击中
            is_hit = False
            hp_reward -= 400
        if has_hit:
            hp_reward -= 100
        hp_reward -= miss_step_count * 0.1
        miss_step_count += 1
    else:
        if not is_hit:  # 第一次击中
            is_hit = True
            has_hit = True
            hit_step_count = 0
            miss_step_count = 0
        hp_reward += 500 + hit_step_count * 10  # 连续命中奖励
        hit_step_count += 1

    reward = 10 * angle_reward + distance_reward + hp_reward
    print('reward=', reward, 'distance_reward=', distance_reward, 'angle_reward=', angle_reward, 'hp_reward=',
          hp_reward)
    return reward
