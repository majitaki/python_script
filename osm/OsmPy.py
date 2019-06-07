import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
from enum import IntEnum, auto
np.set_printoptions(precision=3)



class Env_Dist(IntEnum):
    Normal = auto()
    Binormal = auto()
    Laplace = auto()
    Turara_Fix = auto()
    Turara_Depend_Sensor_Acc = auto()
    
class Agent_Weight_Setting(IntEnum):
    Sensor_Weight_Depend_Sensor_Acc = auto()
    Sensor_Weight_Fix = auto()

class Cheat_Dice_Particle_Filter():
    def __init__(self, sample_num, cheat_rate):
        self.sample_num = sample_num
        self.cheat_rate = cheat_rate
    
    def get_posterior_p(self, prior_p):
        likelihoods = get_likelihoods_for_particle_filter(self.sample_num, prior_p, self.cheat_rate)
        weight_dist = get_weight_distribute_for_particle_filter(prior_p, likelihoods)
        pos_p = prior_p * weight_dist
        
        if np.sum(pos_p) == 0:
            return np.full(self.sample_num, 1.0 / self.sample_num)
        return pos_p / np.sum(pos_p)

		 
class Cheat_Dice_Bayse_Filter():
    def __init__(self, cheat_rate):
        self.cheat_rate = cheat_rate
    
    def get_posterior_p(self, prior_p):
        dim = len(prior_p)
        likelihoods = get_likelihoods_for_particle_filter(dim, prior_p, self.cheat_rate)
        weight_dist = get_weight_distribute_for_particle_filter(prior_p, likelihoods)
        pos_p = prior_p * weight_dist
        
        if np.sum(pos_p) == 0:
            return np.full(dim, 1.0 / dim)
        return pos_p / np.sum(pos_p)



def make_turara(dim, weight):
    max_weight = 1 / (1 + (dim - 1) * (1 - weight))
    other_weight = max_weight * (1 - weight)
    return max_weight, other_weight

def get_likelihoods_for_bayse_filter(dim, sample_index, weight):
    weights = make_turara(dim, weight)
    max_weight = weights[0]
    other_weight = weights[1]
    
    likelihoods = np.full(dim, other_weight)
    likelihoods[sample_index] = max_weight
    return likelihoods

def get_likelihoods_for_particle_filter(dim, sample_array, weight):
    value_exist_num = np.sum(sample_array != 0)
    weights = make_turara(value_exist_num, weight)
    max_weight = weights[0]
    other_weight = weights[1]
    
    max_index = sample_array.argmax()
    likelihoods = np.array([max_weight if i == max_index else
                            other_weight if sample_array[i] != 0 else
                            0
                            for i in range(dim)])
    return likelihoods

def get_weight_distribute_for_particle_filter(sample_array, likelihoods):
    raw_w_dis = sample_array * likelihoods
    return raw_w_dis

def get_pos_p_array_by_bayse_filter(pre_p_array, input_array, weight):
    dim = len(pre_p_array)
    sample_index = input_array.argmax()
    if input_array[sample_index] != 1:
        print("Unexpected")
    likelihoods = get_likelihoods_for_bayse_filter(dim, sample_index, weight)
    pos_p_array = pre_p_array * likelihoods
    if np.sum(pos_p_array) == 0:
        return np.full(dim, 1.0 / dim)
    return pos_p_array / np.sum(pos_p_array)

def get_pos_p_array_by_particle_filter(pre_p_array, input_array, weight):
    dim = len(pre_p_array)
    likelihoods = get_likelihoods_for_particle_filter(dim, input_array, weight)
    weight_dist = get_weight_distribute_for_particle_filter(input_array, likelihoods)
    pos_p_array = pre_p_array * weight_dist
        
    if np.sum(pos_p_array) == 0:
        return np.full(dim, 1.0 / dim)
    return pos_p_array / np.sum(pos_p_array)

def observe_dist(dist_array):
    rnd = random.random()
    accumlation_value = 0
    
    index = 0
    for value in dist_array:
        accumlation_value = accumlation_value + value
        if rnd <= accumlation_value:
            return index
        index = index + 1
    print('error')
    return -1

def multi_observe_dist_to_dataframe(dist_array, num):
    df_input = None
    for i in range(num):
        nd_input = np.zeros(len(dist_array))
        observed_op_index = observe_dist(dist_array)
        nd_input[observed_op_index] = 1
        df_new_input = pd.DataFrame(nd_input)
        if df_input is None:
            df_input = df_new_input
        else:    
            df_input = pd.concat([df_input, df_new_input], axis = 1)
    return df_input

def sensor_simulation(step_size, threshold, op_intro_rate, op_intro_duration, sensor_size, env_array, sensor_acc, belief_setting, samples):
   dim = len(env_array)
   pos_p_array = pre_p_array = np.full(dim, 1.0 / dim)
   sensor_op_array = np.array([0 for d in range(dim)])
   receive_num = 0
   
   for step in range(step_size):
       if(step % op_intro_duration != 0):
           continue
       active_sensor_size = math.ceil(sensor_size * op_intro_rate)
       if(random.random() > active_sensor_size / sensor_size):
           continue
       receive_num = receive_num + 1
       
       if belief_setting == Belief_Setting.BayesFilter:
           df_one_sample = multi_observe_dist_to_dataframe(env_array, 1)
           pos_p_array = get_pos_p_array_by_bayse_filter(pre_p_array, df_one_sample[0].values, sensor_acc)
       elif belief_setting == Belief_Setting.ParticleFilter:
           df_samples = multi_observe_dist_to_dataframe(env_array, samples)
           df_input = df_samples.sum(axis=1)
           pos_p_array = get_pos_p_array_by_particle_filter(pre_p_array, df_input.values, sensor_acc)
       
       pre_p_array = pos_p_array
       
       if(len(np.where(pd.DataFrame(pos_p_array) > threshold)[0]) >= 1):
           sensor_op_array[np.where(pd.DataFrame(pos_p_array) > threshold)[0][0]] = 1
           break
       
   return sensor_op_array, pos_p_array, receive_num


def sensor_simulation_by_step(step_size, threshold, op_intro_rate, op_intro_duration, sensor_size, env_array, sensor_acc, belief_setting, samples, pre_p_array, cur_op_index):
   dim = len(env_array)
   pos_p_array = np.full(dim, 1.0 / dim)
   sensor_op_array = np.array([0 for d in range(dim)])
   receive_num = 0
   opinion_by_steps_array = np.array([-1 for d in range(step_size)])
   current_op_index = cur_op_index
   
   for step in range(step_size):
       opinion_by_steps_array[step] = current_op_index
       if(step % op_intro_duration != 0):
           continue
       active_sensor_size = math.ceil(sensor_size * op_intro_rate)
       if(random.random() > active_sensor_size / sensor_size):
           continue
       receive_num = receive_num + 1
       
       if belief_setting == Belief_Setting.BayesFilter:
           df_one_sample = multi_observe_dist_to_dataframe(env_array, 1)
           pos_p_array = get_pos_p_array_by_bayse_filter(pre_p_array, df_one_sample[0].values, sensor_acc)
       elif belief_setting == Belief_Setting.ParticleFilter:
           df_samples = multi_observe_dist_to_dataframe(env_array, samples)
           df_input = df_samples.sum(axis=1)
           pos_p_array = get_pos_p_array_by_particle_filter(pre_p_array, df_input.values, sensor_acc)
       
       pre_p_array = pos_p_array
       
       if(len(np.where(pd.DataFrame(pos_p_array) > threshold)[0]) >= 1):
           current_op_index = np.where(pd.DataFrame(pos_p_array) > threshold)[0][0]
           opinion_by_steps_array[step] = current_op_index
   
   if current_op_index != -1:
       sensor_op_array[current_op_index] = 1
   return opinion_by_steps_array, pos_p_array, sensor_op_array, receive_num

    
def make_env(env_dist, dim, is_depend_sensor_acc = False, sensor_acc = None, turara_index = 0, is_plot = True):
    env_samples = None
    
    if not is_depend_sensor_acc:
        if env_dist == Env_Dist.Normal:
            mu = 0
            sigma = 1
            size = 1000
            env_samples = np.random.normal(mu , sigma, size)
            
        elif env_dist == Env_Dist.Binormal:
            num = 10
            p = 0.5
            size = 1000
            env_samples = np.random.binomial(num, p, size)
            
        elif env_dist == Env_Dist.Laplace:
            loc, scale = 0.0, 1.5
            size = 1000
            env_samples = np.random.laplace(loc, scale, size)
            
        elif env_dist == Env_Dist.Turara_Fix:
            weights = make_turara(dim, sensor_acc)
            max_weight = weights[0]
            other_weight = weights[1]
            
            env_array = np.full(dim, other_weight)
            env_array[turara_index] = max_weight
            if is_plot:
                pd.DataFrame(env_array).plot(kind = 'bar')
            return env_array
    else:
        if env_dist == Env_Dist.Turara_Depend_Sensor_Acc:
            weights = make_turara(dim, sensor_acc)
            max_weight = weights[0]
            other_weight = weights[1]
            env_array = np.full(dim, other_weight)
            env_array[turara_index] = max_weight
            return env_array
        
    count, bins, ignored = plt.hist(env_samples, dim, normed=True)
    env_array = np.array([(bins[i + 1] - bins[i]) * count[i] for i in range(dim)])
    return env_array

def make_sensor_acc(agent_weight_setting, ori_senror_weight, fix_sensor_weight):
    my_sensor_weight = None
    
    if agent_weight_setting == Agent_Weight_Setting.Sensor_Weight_Depend_Sensor_Acc:
        my_sensor_weight = ori_senror_weight
    elif agent_weight_setting == Agent_Weight_Setting.Sensor_Weight_Fix:
        my_sensor_weight = fix_sensor_weight
    return round(my_sensor_weight, 4)


