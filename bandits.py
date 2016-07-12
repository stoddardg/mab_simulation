from scipy.stats import beta

import numpy as np
import scipy.stats

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
	
	def update(self, arm_id, reward_list):
		self.arm_feedback[arm_id] = self.arm_feedback.get(arm_id,0) + sum(reward_list)
		self.arm_plays[arm_id] = self.arm_plays.get(arm_id,0) + 1.0*(len(reward_list))
		self.arm_mean_payoff[arm_id] = self.arm_feedback[arm_id]/self.arm_plays[arm_id]


class BetaBandit(object):
	def __init__(self, num_options=2, prior =(.5, .5)):
		self.arm_plays = {}
		self.successes = {}
		self.num_options = num_options
		self.prior = prior
	
	def update(self, arm_id, reward_list):

		self.arm_plays[arm_id] = self.arm_plays.get(arm_id, 0) + 1.0*(len(reward_list))

		self.successes[arm_id] = self.successes.get(arm_id, 0)+1.0*(np.sum(reward_list))
	 
	def get_decision(self,arm_id_list,arm_feature_list):
		sampled_theta = []
		for i in arm_id_list:
			dist = beta(self.prior[0]+self.successes.get(i, 0),
				   self.prior[1]+self.arm_plays.get(i,0)-self.successes.get(i,0))
			sampled_theta += [dist.rvs()]
		return sampled_theta.index(max(sampled_theta))   

class UCB():
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
	
	def update(self, arm_id, reward_list):
			self.total_plays += 1
			self.arm_plays[arm_id] = 1.0*len(reward_list) + self.arm_plays.get(arm_id,0)
			self.arm_rewards[arm_id] = reward + self.arm_rewards.get(arm_id, 0)


	
		