import numpy as np

from collections import Counter
def evaluate_bandit(sim, mab, time_steps):
    reward_list = []
    arm_play_counter = Counter() 
    for t in np.arange(time_steps):
        arms, arm_features = sim.get_available_arms()
        arm_to_play = mab.get_decision(arms, arm_features)
        
        arm_play_counter[arm_to_play] += 1.0
        
        # reward is itself a list of successes and failures drawn 
        # from a Bernouilli distribution
        reward = sim.get_reward(arm_to_play)
        mab.update(arm_to_play, reward)
        reward_list.append(reward*1.0)
    
    total_reward = np.sum(reward_list)
    average_reward_by_time = np.cumsum(reward_list)/np.arange(1,time_steps+1.0)
    
    return total_reward, average_reward_by_time