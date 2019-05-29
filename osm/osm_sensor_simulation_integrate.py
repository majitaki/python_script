import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from enum import IntEnum, auto

def convert_p(pre_p, pre_p_dim, rcv_dim, base, dim_size, is_rcv_new):
  if not is_rcv_new:
      return pre_p

  if pre_p_dim == rcv_dim:
      return 1.0 / (base + dim_size)
  else:
      return pre_p / (1.0 + 1.0 / (base + dim_size))


def convert_w(w, pre_p_dim, rcv_dim, dim_size, float_len):
   if float_len != 0.0:
       w = (w - 1 / dim_size) * float_len + 1 / dim_size
       #w = w * float_len

   if pre_p_dim == rcv_dim:
       return w
   else:
      return (1 - w)/(dim_size - 1)



def is_new_dim(pre_p_list, rcv_vector):
  if len(pre_p_list) < len(rcv_vector):
     return True
  else:
     return False


def get_pos_p_list_with_bayes_filter(pre_p_list, rcv_vector, w):
  base = 100
  is_rcv_new = False

  for rcv_dim, rcv_op_length in enumerate(rcv_vector):
      if len(pre_p_list) < len(rcv_vector):
          is_rcv_new = True
          pre_p_list.append(0.0)

      float_len, int_len = math.modf(rcv_op_length)
      for i in range(int(int_len)):
          pos_p_list = []
          for pre_p_dim, pre_p in enumerate(pre_p_list):
              upper = convert_p(pre_p, pre_p_dim, rcv_dim, base, len(pre_p_list), is_rcv_new) * convert_w(w, pre_p_dim, rcv_dim, len(pre_p_list), 0.0)
              lower = 0.0

              for pre_p_sub_dim, pre_p_sub in enumerate(pre_p_list):
                  tmp = convert_p(pre_p_sub, pre_p_sub_dim, rcv_dim, base, len(pre_p_list), is_rcv_new) * convert_w(w, pre_p_sub_dim, rcv_dim, len(pre_p_list), 0.0)
                  lower = lower + tmp

              pos_p = round(upper / lower, 4)
              pos_p_list.append(pos_p)
          pre_p_list = pos_p_list
          is_rcv_new = False

      if float_len != 0.0:
          pos_p_list = []
          for pre_p_dim, pre_p in enumerate(pre_p_list):
              upper = convert_p(pre_p, pre_p_dim, rcv_dim, base, len(pre_p_list), is_rcv_new) * convert_w(w, pre_p_dim, rcv_dim, len(pre_p_list), float_len)
              lower = 0.0

              for pre_p_sub_dim, pre_p_sub in enumerate(pre_p_list):
                  tmp = convert_p(pre_p_sub, pre_p_sub_dim, rcv_dim, base, len(pre_p_list), is_rcv_new) * convert_w(w, pre_p_sub_dim, rcv_dim, len(pre_p_list), float_len)
                  lower = lower + tmp

              pos_p = round(upper / lower, 4)
              pos_p_list.append(pos_p)
          pre_p_list = pos_p_list

  return pre_p_list


def extract_op(env_info_distribute):
    rnd = random.random()
    accumlation_belief = 0
    for index, env_info_value in env_info_distribute.iteritems():
        accumlation_belief = accumlation_belief + env_info_value
        if rnd <= accumlation_belief:
            return index
    return -1

def sensor_simulation(step_size, threshold, op_intro_rates, op_intro_duration, sensor_size, env_info_distribute, sensor_acc):
   dim = len(env_info_distribute.index)
   pre_p_list = [1.0 / dim for x in range(dim)]
   #df = pd.DataFrame()
   #df = df.append(pd.Series(pre_p_list), ignore_index = True)
   sensor_op = [0 for d in range(dim)]

   for step in range(step_size):
       if(step % op_intro_duration != 0):
           continue
       active_sensor_size = math.ceil(sensor_size * op_intro_rates)
       if(random.random() > active_sensor_size / sensor_size):
           continue

       rcv_vector = [0 for d in range(dim)]
       rcv_index = extract_op(env_info_distribute)
       rcv_vector[rcv_index] = 1
      
       pos_p_list = get_pos_p_list_with_bayes_filter(pre_p_list, rcv_vector, sensor_acc)
       pre_p_list = pos_p_list
       
       if(len(np.where(pd.DataFrame(pre_p_list) > threshold)[0]) >= 1):
           sensor_op[np.where(pd.DataFrame(pre_p_list) > threshold)[0][0]] = 1
           break
   return sensor_op, pre_p_list

class Env_Dist(IntEnum):
    Normal = auto()
    Binormal = auto()
    Laplace = auto()
    Turara_Fix = auto()
    Turara_Depend_Sensor_Acc = auto()
    
class Agent_Weight_Setting(IntEnum):
    Sensor_Weight_Depend_Sensor_Acc = auto()
    Sensor_Weight_Fix = auto()
    
    
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
            env_p = []
            most_r = fix_turara
            other_r = (1 - most_r) / (dim - 1)
            
            for d in range(dim):
                if(d == 0):
                    env_p.append(most_r)
                else:
                    env_p.append(other_r)
            env_df = pd.DataFrame(env_p)
            env_df.plot(kind = 'bar')
            return env_df
    else:
        if env_dist == Env_Dist.Turara_Depend_Sensor_Acc:
            env_p = []
            most_r = sensor_acc
            other_r = (1 - most_r) / (dim - 1)
            
            for d in range(dim):
                if(d == 0):
                    env_p.append(most_r)
                else:
                    env_p.append(other_r)
            env_df = pd.DataFrame(env_p)
            #env_df.plot(kind = 'bar')
            return env_df
        
    count, bins, ignored = plt.hist(env_samples, dim, normed=True)
    
    env_p = []
    for i in range(dim):
        diff_bin = bins[i + 1] - bins[i]
        p = diff_bin * count[i]
        env_p.append(p)
    
    env_df = pd.DataFrame(env_p)
    return env_df

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
dim = 2
step_size = 3000
rounds = 5
threshold = 0.90
sensor_size = 30
op_intro_rates = 0.1
op_intro_duration = 10
op_intro_size = 3 * sensor_size
multi = True
duration_sensor_rate = 0.05

graph_columns = ['sensor_acc', 'correct_rate', 'incorrect_rate', 'undeter_rate']
df_desc_acc = pd.DataFrame(columns = graph_columns)

env_df = None
thete_map = None

if my_env_dist == Env_Dist.Turara_Fix:
    env_df = make_env(my_env_dist, dim, False, fix_turara=my_fix_turara )
    thete_map = env_df.idxmax()

for sensor_acc in np.arange(1/dim, 1.0, duration_sensor_rate):
   rounds_correct_sensor_rate_list =[]
   rounds_undeter_sensor_rate_list =[]
   rounds_incorrect_sensor_rate_list =[]

   for r in tqdm(range(rounds), desc = "Sensor Acc: " + str(round(sensor_acc, 4))):
       df_output_op = pd.DataFrame()

       if my_env_dist == Env_Dist.Turara_Depend_Sensor_Acc:
           env_df = make_env(my_env_dist, dim, True, sensor_acc = sensor_acc)
           thete_map = env_df.idxmax()
           
       my_sensor_acc = make_sensor_acc(my_agent_weight_setting, sensor_acc, my_fix_sensor_acc)

       if(multi):
           results = Parallel(n_jobs = -1)(
                   [delayed(sensor_simulation)(step_size, threshold, op_intro_rates, op_intro_duration, sensor_size, env_df[0], my_sensor_acc)
                   for j in range(sensor_size)]
                   )

           for result in results:
               df_output_op = df_output_op.append(pd.DataFrame(result[0]).T)
       else:
           for i in range(sensor_size):
               output_op = sensor_simulation(step_size, threshold, op_intro_rates, op_intro_duration, sensor_size, env_df[0], my_sensor_acc)
               df_output_op = df_output_op.append(pd.DataFrame(output_op).T)


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
