import numpy as np

from collections import Counter

def eval_bandit(sim, mab, time_steps, plays_per_time_step, TRIALS=1, 
                warm_start=False,plays_warm_start=100,n_arms_warm_start=10, random_seed=None):
    
    total_avg_reward_list = np.zeros(time_steps)
    total_reward = 0
    for trial in np.arange(TRIALS):
        if random_seed is not None:
            rs = random_seed*trial
        else:
            rs = None
        sim.reset(random_seed=rs)
        mab.reset()
        reward_list = []

        if warm_start == True:
            arms, arm_features = sim.get_available_arms()
            random_arm_list = np.random.choice(arms, replace=False, size=n_arms_warm_start)
            for arm in random_arm_list:
                reward = sim.get_reward(arm, n_pulls=plays_warm_start)
                arm_features = sim.get_arm_features(arm)
                mab.update(arm, arm_features, reward)



        for t in np.arange(time_steps):

            arms, arm_features = sim.get_available_arms()
            arm_to_play = mab.get_decision(arms, arm_features)
            
         
            reward = sim.get_reward(arm_to_play, n_pulls = plays_per_time_step)

            arm_features = sim.get_arm_features(arm_to_play)

            mab.update(arm_to_play, arm_features, reward)
            reward_list.append(np.sum(reward)*1.0)
        
        running_avg_reward = np.cumsum(reward_list)/np.arange(1,time_steps+1.0)
        running_avg_reward /= plays_per_time_step*max(sim.arm_probs.values())

        total_avg_reward_list += running_avg_reward
        total_reward += np.sum(reward_list) / (plays_per_time_step*time_steps*max(sim.arm_probs.values()))

    total_reward /= (1.0*TRIALS)

    total_avg_reward_list /= (TRIALS)

    return total_reward, total_avg_reward_list








