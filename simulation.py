from scipy.stats import beta

import numpy as np
import scipy.stats

from collections import Counter
import sys

# Scales the arm_probs to all be between 0 and 1.
# The highest pre-scaled value is assigned a value of 1.
# Probs is a dictionary
def _scale(probs, lower_bound, upper_bound):
    # min_int
    biggest = - sys.maxint - 1
    interval = upper_bound - lower_bound
    
    for i in range(len(probs)): 
        if abs(probs[i]) > biggest:
            biggest = abs(probs[i])
    biggest = biggest * 2 / interval
    for i in range(len(probs)):
        probs[i] = probs[i]/biggest + (.5*(upper_bound + lower_bound))
    
    return probs
    

class SimpleSimulator:
    def __init__(self, n_arms, loc_value, scale_value):
        self.arm_probs = {}
        self.arm_features = []
        temp_val_list = []
        scaled_vals = []

        for i in range(0, n_arms):
            temp_val = np.random.normal(loc=loc_value, scale = scale_value) 
            temp_val_list.append(temp_val)
        scaled_vals = _scale(temp_val_list, lower_bound=0, upper_bound=.2)
        for i in range(0,n_arms):
            self.arm_probs[i] = scaled_vals[i] 

    def get_available_arms(self):
        return self.arm_probs.keys(), []
    
    def get_reward(self, arm_id):
        return scipy.stats.bernoulli.rvs(self.arm_probs.get(arm_id,0))
    
    def get_arm_features(self, arm_id):
        return self.arm_features

    def exists():
        print "exists"




