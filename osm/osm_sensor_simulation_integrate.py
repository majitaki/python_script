import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from enum import IntEnum, auto

class Env_Dist(IntEnum):
    Normal = auto()
    Binormal = auto()
    Laplace = auto()
    Turara_Fix = auto()
    Turara_Depend_Sensor_Acc = auto()
    
class Agent_Weight_Setting(IntEnum):
    Sensor_Weight_Depend_Sensor_Acc = auto()
    Sensor_Weight_Fix = auto()

class Belief_Setting(IntEnum):
    BayesFilter = auto()
    ParticleFilter = auto()

def get_likelihoods_for_bayse_filter(dim, sample_index, weight):
    max_weight = weight
    other_weight = (1 - max_weight) / (dim - 1)
    likelihoods = np.full(dim, other_weight)
    likelihoods[sample_index] = max_weight
    return likelihoods / np.sum(likelihoods)

def get_likelihoods_for_particle_filter(dim, sample_array, weight):
    value_exist_num = np.sum(sample_array != 0)
    max_weight = weight
    other_weight = (1 - max_weight) / (value_exist_num - 1)
    max_index = sample_array.argmax()
    likelihoods = np.array([max_weight if i == max_index else
                            other_weight if sample_array[i] != 0 else
                            0
                            for i in range(dim)])
    
    return likelihoods / np.sum(likelihoods)

def get_weight_distribute_for_particle_filter(sample_array, likelihoods):
    raw_w_dis = sample_array * likelihoods
    return raw_w_dis / np.sum(raw_w_dis)

def get_pos_p_array_by_bayse_filter(pre_p_array, df_input, weight):
    dim = len(pre_p_array)
    for clm, item in df_input.iteritems():
        sample_index = item.idxmax()
        if item[sample_index] != 1:
            print("Unexpected")
        
        likelihoods = get_likelihoods_for_bayse_filter(dim, sample_index, weight)
        pos_p_array = pre_p_array * likelihoods
        pre_p_array = pos_p_array
    return pos_p_array / np.sum(pos_p_array)

def get_pos_p_array_by_particle_filter(pre_p_array, df_input, weight, sample_num):
    dim = len(pre_p_array)
    for item in df_input.iteritems():
        likelihoods = get_likelihoods_for_particle_filter(dim, item.values, weight)
        weight_dist = get_weight_distribute_for_particle_filter(item.values, likelihoods)
        pos_p_array = pre_p_array * weight_dist
        pre_p_array = pos_p_array
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
        nd_input = np.zeros(dim)
        observed_op_index = observe_dist(dist_array)
        nd_input[observed_op_index] = 1
        df_new_input = pd.DataFrame(nd_input)
        if df_input == None:
            df_input = df_new_input
        else:    
            df_input = pd.concat([df_input, df_new_input], axis = 1)
    return df_input

def sensor_simulation(step_size, threshold, op_intro_rate, op_intro_duration, sensor_size, env_array, sensor_acc, belief_setting, samples):
   dim = len(env_array)
   pre_p_array = np.full(dim, 1.0 / dim)
   sensor_op_array = np.array([0 for d in range(dim)])
   df_input = None
   
   for step in range(step_size):
       if(step % op_intro_duration != 0):
           continue
       active_sensor_size = math.ceil(sensor_size * op_intro_rate)
       if(random.random() > active_sensor_size / sensor_size):
           continue
       
       if belief_setting == Belief_Setting.BayesFilter:
           df_input = multi_observe_dist_to_dataframe(env_array, 1)
           pos_p_array = get_pos_p_array_by_bayse_filter(pre_p_array, df_input, sensor_acc)
       elif belief_setting == Belief_Setting.ParticleFilter:
           df_input = multi_observe_dist_to_dataframe(env_array, samples)
           pos_p_array = get_pos_p_array_by_particle_filter(pre_p_array, df_input, sensor_acc)
       
       pre_p_array = pos_p_array
       
       if(len(np.where(pd.DataFrame(pos_p_array) > threshold)[0]) >= 1):
           sensor_op_array[np.where(pd.DataFrame(pos_p_array) > threshold)[0][0]] = 1
           break
       
   return sensor_op_array, pos_p_array

    
def make_env(env_dist, dim, is_depend_sensor_acc = False, sensor_acc = None, fix_turara = None):
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
            max_weight = fix_turara
            other_weight = (1 - max_weight) / (dim - 1)
            env_array = np.full(dim, other_weight)
            env_array[0] = max_weight
            pd.DataFrame(env_array).plot(kind = 'bar')
            return env_array
    else:
        if env_dist == Env_Dist.Turara_Depend_Sensor_Acc:
            max_weight = sensor_acc
            other_weight = (1 - max_weight) / (dim - 1)
            env_array = np.full(dim, other_weight)
            env_array[0] = max_weight
            return env_array
        
    count, bins, ignored = plt.hist(env_samples, dim, normed=True)
    env_array = [(bins[i + 1] - bins[i]) * count[i] for i in range(dim)]
    return env_array

def make_sensor_acc(agent_weight_setting, ori_senror_acc, fix_sensor_acc):
    my_sensor_acc = None
    
    if agent_weight_setting == Agent_Weight_Setting.Sensor_Weight_Depend_Sensor_Acc:
        my_sensor_acc = ori_senror_acc
    elif agent_weight_setting == Agent_Weight_Setting.Sensor_Weight_Fix:
        my_sensor_acc = fix_sensor_acc
    return my_sensor_acc


my_env_dist = Env_Dist.Turara_Depend_Sensor_Acc
my_fix_turara = 0.2
my_agent_weight_setting = Agent_Weight_Setting.Sensor_Weight_Depend_Sensor_Acc
my_fix_sensor_acc = 0.7
my_belief_setting = Belief_Setting.BayesFilter
my_samples = 10

dim = 2
step_size = 1500
rounds = 5
threshold = 0.90
sensor_size = 30
op_intro_rate = 0.1
op_intro_duration = 10
op_intro_size = 3 * sensor_size
multi = True
duration_sensor_rate = 0.05

graph_columns = ['sensor_acc', 'correct_rate', 'incorrect_rate', 'undeter_rate']
df_desc_acc = pd.DataFrame(columns = graph_columns)

env_array = None
thete_map = None

if my_env_dist == Env_Dist.Turara_Fix:
    env_array = make_env(my_env_dist, dim, False, fix_turara = my_fix_turara)
    thete_map = env_array.argmax()

for sensor_acc in np.arange(1/dim, 1.0, duration_sensor_rate):
   rounds_correct_sensor_rate_list =[]
   rounds_undeter_sensor_rate_list =[]
   rounds_incorrect_sensor_rate_list =[]

   for r in tqdm(range(rounds), desc = "Sensor Acc: " + str(round(sensor_acc, 4))):
       df_output_op = pd.DataFrame()

       if my_env_dist == Env_Dist.Turara_Depend_Sensor_Acc:
           env_array = make_env(my_env_dist, dim, True, sensor_acc = sensor_acc)
           thete_map = env_array.argmax()
           
       my_sensor_acc = make_sensor_acc(my_agent_weight_setting, sensor_acc, my_fix_sensor_acc)

       results = sensor_simulation(step_size, threshold, op_intro_rate, op_intro_duration, sensor_size, env_array, my_sensor_acc, my_belief_setting, my_samples)

       results = Parallel(n_jobs = -1)(
               [delayed(sensor_simulation)(step_size, threshold, op_intro_rate, op_intro_duration, sensor_size, env_array, my_sensor_acc, my_belief_setting, my_samples)
               for j in range(sensor_size)]
               )
       for result in results:
           df_output_op = df_output_op.append(pd.DataFrame(result[0]).T)    

       df_correct = pd.DataFrame(df_output_op[thete_map] == 1)
       df_undeter = pd.DataFrame(df_output_op.sum(axis = 1))
       correct_sensor_rate = df_correct.where(df_correct == True).dropna().count() / sensor_size
       undeter_sensor_rate = df_undeter.where(df_undeter == 0).dropna().count()[0] / sensor_size
       incorrect_sensor_rate = 1.0 - correct_sensor_rate - undeter_sensor_rate

       rounds_correct_sensor_rate_list.append(correct_sensor_rate.values[0])
       rounds_undeter_sensor_rate_list.append(undeter_sensor_rate)
       rounds_incorrect_sensor_rate_list.append(incorrect_sensor_rate.values[0])


   desc_acc = [sensor_acc, round(np.average(rounds_correct_sensor_rate_list), 4), round(np.average(rounds_incorrect_sensor_rate_list), 4), round(np.average(rounds_undeter_sensor_rate_list), 4)]
   df_desc_acc = df_desc_acc.append(pd.DataFrame([desc_acc], columns = graph_columns))

plt.figure()
colorlist = ["g", "r", "grey"]
ax = df_desc_acc.plot(title = 'DimSize:' + str(dim) + ' StepSize:' + str(step_size),
             kind = 'area',
             x = df_desc_acc.columns[0],
             color = colorlist,
             alpha=0.5,
             figsize=(8,5)
             )

plt.ylim([0, 1.0])
plt.xlim([0, 1.0])
plt.xticks(np.arange(0.0, 1.0, 0.05))
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.xlabel("Sensor Acc", fontsize=18)
plt.ylabel("Sensor Opinion Acc", fontsize=18)
plt.tick_params(labelsize=8)
plt.grid(True)
