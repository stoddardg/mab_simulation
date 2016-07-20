import numpy as np

from collections import Counter

def evaluate_bandit(sim, mab, TIME_STEPS, plays_per_time_step):
    reward_list = []
    arm_play_counter = Counter() 
    for t in np.arange(time_steps):
        arms, arm_features = sim.get_available_arms()
        arm_to_play = mab.get_decision(arms, arm_features)
        
        arm_play_counter[arm_to_play] += 1.0
     
        reward = sim.get_reward(arm_to_play, n_pulls = plays_per_time_step)

        arm_features = sim.get_arm_features(arm_to_play)

        mab.update(arm_to_play, arm_features, reward)
        reward_list.append(np.sum(reward)*1.0)
    
    total_reward = np.sum(reward_list)
    average_reward_by_time = np.cumsum(reward_list)/np.arange(1,time_steps+1.0)
    
    return total_reward, average_reward_by_time