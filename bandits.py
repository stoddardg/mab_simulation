from scipy.stats import beta

import numpy as np
import scipy.stats

from sklearn import linear_model
from sklearn.base import clone
from sklearn.metrics import log_loss

from sklearn.feature_extraction import FeatureHasher


from collections import Counter

class EGreedyMAB:
    
    def __init__(self, epsilon=.1):
        self.arm_feedback = {}
        self.arm_plays = {}
        self.arm_mean_payoff = {}
        self.epsilon = epsilon
        
        
    def get_decision(self,arm_id_list, arm_feature_list):
        np.random.shuffle(arm_id_list)
        current_averages = {id: self.arm_mean_payoff.get(id,100) for id in arm_id_list}
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(arm_id_list)
        else:
            return max(current_averages, key=current_averages.get)
    
    def update(self, arm_id, features, reward_list):
        self.arm_feedback[arm_id] = self.arm_feedback.get(arm_id,0) + sum(reward_list)
        self.arm_plays[arm_id] = self.arm_plays.get(arm_id,0) + 1.0*(len(reward_list))
        self.arm_mean_payoff[arm_id] = self.arm_feedback[arm_id]/self.arm_plays[arm_id]

    def reset(self):
        self.__init__(epsilon=self.epsilon)


        
class BetaBandit:
    def __init__(self, prior =(1, 1)):
        self.arm_plays = {}
        self.successes = {}
        self.prior = prior
    
    def update(self, arm_id, features, reward_list):
        self.arm_plays[arm_id] = self.arm_plays.get(arm_id, 0) + 1.0*(len(reward_list))
        self.successes[arm_id] = self.successes.get(arm_id, 0)+1.0*(np.sum(reward_list))
     
    def get_decision(self,arm_id_list,arm_feature_list):
        successes = np.array([self.successes.get(i,0) for i in arm_id_list])
        plays = np.array([self.arm_plays.get(i,0) for i in arm_id_list])
        fails = plays - successes
        
        alpha_vals = self.prior[0] + successes
        beta_vals = self.prior[1] + fails
        
        dist = beta(alpha_vals, beta_vals)
        sampled_theta_list = dist.rvs()
        return arm_id_list[np.argmax(sampled_theta_list)]

    def reset(self):
        self.__init__(prior=self.prior)
class UCB:
    def __init__(self):
        self.arm_plays = {}
        self.arm_rewards = {}
        self.total_plays = 1
    
    def get_decision(self,arm_id_list,arm_feature_list):
        ucb_values = {}
        for arm in arm_id_list:
            bonus = math.sqrt((2*math.log(self.total_plays))/float(self.arm_plays.get(arm, 1)))
            ucb_values[arm] = (self.arm_rewards.get(arm, 0)/self.arm_plays.get(arm, 1)) + bonus
        return max(ucb_values, key=ucb_values.get)
    
    def update(self, arm_id, features, reward_list):
            self.total_plays += 1.0*len(reward_list)
            self.arm_plays[arm_id] = 1.0*len(reward_list) + self.arm_plays.get(arm_id,0)
            self.arm_rewards[arm_id] = 1.0*(np.sum(reward_list)) + self.arm_rewards.get(arm_id, 0)

    def reset(self):
        self.__init__()



def hash_features(features, arm_ids, use_id=True):
    n_features = np.shape(features)[1]
    feature_names = [str(x) for x in np.arange(n_features)]
    all_features = []
    for arm_id, feature_set in zip(arm_ids, features):
        temp_features = zip(feature_names, feature_set)
        if use_id == True:
            temp_features.append(("id_"+str(arm_id), 1))
        all_features.append(temp_features)

    f = FeatureHasher(input_type='pair')
    return f.transform(all_features)


class EGreedy_Logistic:
    def __init__(self, clf=None,fixed_effects=False, epsilon=0, alpha=.0001):
        self.alpha = alpha
        self.is_fit = False
        self.fixed_effects = fixed_effects
        self.epsilon = epsilon
        if clf is None:
            self.clf = linear_model.SGDClassifier(loss='log',alpha=self.alpha)
        else:
            self.clf = clone(clf)
    
    def update(self, arm_id, arm_features, rewards):
        # For now, we will ignore arm_id because its a pain in the ass
        X_features = np.tile(arm_features, [len(rewards),1])
#         X_features = np.tile(1, [len(rewards), 1])
        X_id = np.tile(arm_id, len(rewards))
        
        if self.fixed_effects == True:        
            X_train = hash_features(X_features, X_id, use_id=self.fixed_effects)
        else:
            X_train = X_features
        
        self.clf.partial_fit(X_train, rewards, classes=[0,1])
        self.is_fit = True

    ## Assuming that the data is in the same format as the training data
    def get_decision(self, arm_id_list, arm_feature_list):
        if self.is_fit == False:
            return np.random.choice(arm_id_list)
        if np.random.rand() < self.epsilon:
            return np.random.choice(arm_id_list)

        if self.fixed_effects == True:
            X_data = hash_features(arm_feature_list, arm_id_list, use_id=self.fixed_effects)
        else:
            X_data = arm_feature_list
        probs =  [x[1] for x in self.clf.predict_proba(X_data)]
        return arm_id_list[np.argmax(probs)]                
        

    def reset(self):
        self.__init__(epsilon=self.epsilon, fixed_effects=self.fixed_effects, alpha=self.alpha)
