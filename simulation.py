from scipy.stats import beta

import numpy as np
import scipy.stats

from collections import Counter
from scipy.special import expit

    

class SimpleSimulator:
    def __init__(self, n_arms, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.arm_probs = {}
        self.arm_features = []
        temp_val_list = []
        scaled_vals = []

        self.arm_probs = beta.rvs(1,1,size=n_arms)


    def get_available_arms(self):
        return self.arm_probs.keys(), []
    
    def get_reward(self, arm_id):
        return scipy.stats.bernoulli.rvs(self.arm_probs.get(arm_id,0))
    
    def get_arm_features(self, arm_id):
        return self.arm_features

    def exists():
        print "exists"

    def reset(self, random_seed=None):
        self.__init__(len(self.arm_probs), random_seed=random_seed)


class DisjointLogisticSimulator:
    
    def __init__(self,n_arms,n_features=100, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.arm_probs = {}
        self.arm_features = {}
        self.arm_betas = {}
        self.n_features = n_features
        
        temp_features = scipy.stats.norm.rvs(0,1,size=[n_arms, n_features])
        temp_betas = scipy.stats.norm.rvs(0,.1,size=[n_arms, n_features])
        counter = 0
        for feature_vector, beta_vector in zip(temp_features,temp_betas):
            self.arm_features[counter] = feature_vector
            self.arm_betas[counter] = beta_vector
            temp_arm_prob = expit(np.dot(beta_vector, feature_vector))
            self.arm_probs[counter] = temp_arm_prob
            counter += 1
        
    def get_available_arms(self):
        return self.arm_features.keys(), self.arm_features.values()
    
    def get_arm_features(self, arm_id):
        return self.arm_features.get(arm_id,[])
    
    def get_reward(self, arm_id, n_pulls=1):
        return scipy.stats.bernoulli.rvs(self.arm_probs.get(arm_id,0), size=n_pulls)

    def reset(self, random_seed=None, n_arms=None):
        if self.n_arms is None:
            new_arms = len(self.arm_probs)
        else:
            new_arms = n_arms
        self.__init__(new_arms, n_features=self.n_features, random_seed=random_seed)

    def remove_arms(self, arm_id_list):
        for arm_id in arm_id_list:
            if arm_id in self.arm_probs:
                del self.arm_probs[arm_id]
            if arm_id in self.arm_features:
                del self.arm_features[arm_id]
            if arm_id in self.arm_betas:
                del self.arm_betas[arm_id] 


class GlobalLogisticSimulator:
    
    def __init__(self,n_arms,n_features=100, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.arm_probs = {}
        self.arm_features = {}
        self.beta_vector = scipy.stats.norm.rvs(0,1,size=n_features)
        self.n_features = n_features


        temp_features = scipy.stats.norm.rvs(0,.1,size=[n_arms, n_features])
        
        counter = 0
        for feature_vector in temp_features:
            self.arm_features[counter] = feature_vector
            temp_arm_prob = expit(np.dot(self.beta_vector, feature_vector))
            self.arm_probs[counter] = temp_arm_prob
            counter += 1
        
    def get_available_arms(self):
        return self.arm_features.keys(), self.arm_features.values()
    
    def get_arm_features(self, arm_id):
        return self.arm_features.get(arm_id,[])
    
    def get_reward(self, arm_id, n_pulls=1):
        return scipy.stats.bernoulli.rvs(self.arm_probs.get(arm_id,0), size=n_pulls)
    
    def reset(self, random_seed=None, n_arms=None):
        if n_arms is None:
            new_arms = len(self.arm_probs)
        else:
            new_arms = n_arms
        self.__init__(new_arms, n_features=self.n_features, random_seed=random_seed)

    def remove_arms(self, arm_id_list):
        for arm_id in arm_id_list:
            if arm_id in self.arm_probs:
                del self.arm_probs[arm_id]
            if arm_id in self.arm_features:
                del self.arm_features[arm_id]

