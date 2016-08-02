import numpy as np

from collections import Counter

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def eval_bandit(sim, mab, time_steps, plays_per_time_step, TRIALS=1, 
                warm_start=False,plays_warm_start=100,n_arms_warm_start=10, random_seed=None):
    
    total_avg_reward_list = np.zeros(time_steps)
    total_reward = 0

    n_arms = len(sim.arm_probs)

    if warm_start == True:
        n_arms -= n_arms_warm_start

    total_arms = len(sim.arm_probs)
    arm_play_data = np.zeros(n_arms)

    total_reward_list = []

    for trial in np.arange(TRIALS):
        if random_seed is not None:
            rs = random_seed*trial
        else:
            rs = None
        sim.reset(random_seed=rs, n_arms=total_arms)
        mab.reset()
        reward_list = []

        if warm_start == True:
            arms, arm_features = sim.get_available_arms()
            random_arm_list = np.random.choice(arms, replace=False, size=n_arms_warm_start)
            for arm in random_arm_list:
                reward = sim.get_reward(arm, n_pulls=plays_warm_start)
                arm_features = sim.get_arm_features(arm)
                mab.update(arm, arm_features, reward)
            sim.remove_arms(random_arm_list)
            mab.remove_arms(random_arm_list)


        for t in np.arange(time_steps):

            arms, arm_features = sim.get_available_arms()



            arm_to_play = mab.get_decision(arms, arm_features)
            
         
            reward = sim.get_reward(arm_to_play, n_pulls = plays_per_time_step)

            arm_features = sim.get_arm_features(arm_to_play)

            mab.update(arm_to_play, arm_features, reward)
            reward_list.append(np.sum(reward)*1.0)
        


        probs = sim.arm_probs
        plays = mab.arm_plays
        arm_data = [ (probs.get(x,0), plays.get(x,0)) for x in probs.keys()]
        sorted_arm_data = sorted(arm_data, key = lambda x: x[0])

        normalized_arm_plays = np.array([x[1] for x in sorted_arm_data]) / (1.0*time_steps*plays_per_time_step)

        # print normalized_arm_plays

        arm_play_data += normalized_arm_plays



        running_avg_reward = np.cumsum(reward_list)/np.arange(1,time_steps+1.0)
        running_avg_reward /= plays_per_time_step*max(sim.arm_probs.values())

        total_avg_reward_list += running_avg_reward
        


        total_reward = np.sum(reward_list) / (plays_per_time_step*time_steps*max(sim.arm_probs.values()))

        total_reward_list.append(total_reward)


    avg_reward, lower, upper = mean_confidence_interval(total_reward_list)

    # total_reward /= (1.0*TRIALS)

    total_avg_reward_list /= (TRIALS)

    arm_play_data /= (TRIALS)

    return avg_reward, (lower, upper),  total_avg_reward_list, arm_play_data








