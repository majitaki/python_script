import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
from tqdm import tqdm
from time import sleep
from joblib import Parallel, delayed

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


def get_pos_p_list(pre_p_list, rcv_vector, w):
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

def sensor_simulation(step_size, sensor_rates, threshold, op_intro_rates, op_intro_duration, sensor_size):
    dim = len(sensor_rates)
    pre_p_list = [1.0 / dim for x in range(dim)]
    #df = pd.DataFrame()
    #df = df.append(pd.Series(pre_p_list), ignore_index = True)
    sensor_op = [0 for d in range(dim)]
    w = sensor_rates[0]
    w = 0.7
    
    for step in range(step_size):
        if(step % op_intro_duration != 0):
            continue
        active_sensor_size = math.ceil(sensor_size * op_intro_rates)
        if(random.random() > active_sensor_size / sensor_size):
            continue
        #else:
            #print('step: ' + str(step) + ' observe')
            
        rcv_vector = [0 for d in range(dim)]
        if(random.random() <= sensor_rates[0]):
            rcv_vector[0] = 1 
            #w = sensor_rates[0]
        else:
            rand_dim = random.randrange(1,dim)
            rcv_vector[rand_dim] = 1
            #w = sensor_rates[rand_dim]
            
        #print('- message:' + str(rcv_vector))
        #print('- w:' + str(round(w, 4)))
            
        pos_p_list = get_pos_p_list(pre_p_list, rcv_vector, w)
        #print('- pre:' + str(pre_p_list) + '-> pos:' + str(pos_p_list))
        pre_p_list = pos_p_list
        #df = df.append(pd.Series(pre_p_list), ignore_index = True)
        
        if(len(np.where(pd.DataFrame(pre_p_list) > threshold)[0]) >= 1):
            sensor_op[np.where(pd.DataFrame(pre_p_list) > threshold)[0][0]] = 1
            #print('- op:' + str(sensor_op))
            break
    return sensor_op


dim = 100
step_size = 1500
rounds = 1
threshold = 0.90
sensor_size = 30
op_intro_rates = 0.1
op_intro_duration = 10
op_intro_size = 3 * sensor_size
multi = True
duration_sensor_rate = 0.02
    
df_desc_acc = pd.DataFrame(columns=['sensor_rate', 'correct_rate', 'incorrect_rate', 'undeter_rate'])

for sensor_rate in np.arange(1/dim, 1.0, duration_sensor_rate):
    sensor_rates = []    
    most_r = sensor_rate
    other_r = (1 - most_r) / (dim - 1)
    
    for d in range(dim):
        if(d == 0):
            sensor_rates.append(most_r)
        else:
            sensor_rates.append(other_r)
    
    rounds_correct_sensor_rate_list =[]
    rounds_undeter_sensor_rate_list =[]
    rounds_incorrect_sensor_rate_list =[]
    
    for r in tqdm(range(rounds), desc = "Sensor Rate: " + str(round(most_r, 4))):
        df_output_op = pd.DataFrame()
        
        if(multi):
            results = Parallel(n_jobs = -1)(
                    [delayed(sensor_simulation)(step_size, sensor_rates, threshold, op_intro_rates, op_intro_duration, sensor_size)
                    for j in range(sensor_size)]
                    )
            
            for result in results:
                df_output_op = df_output_op.append(pd.DataFrame(result).T)   
        else:  
            for i in range(sensor_size):            
                output_op = sensor_simulation(step_size, sensor_rates, threshold, op_intro_rates, op_intro_duration, sensor_size)
                df_output_op = df_output_op.append(pd.DataFrame(output_op).T)    
        
        
        df_correct = pd.DataFrame(df_output_op[0] == 1)
        df_undeter = pd.DataFrame(df_output_op.sum(axis = 1))
        correct_sensor_rate = df_correct.where(df_correct == True).dropna().count()[0] / sensor_size 
        undeter_sensor_rate = df_undeter.where(df_undeter == 0).dropna().count()[0] / sensor_size
        incorrect_sensor_rate = 1.0 - correct_sensor_rate - undeter_sensor_rate
        
        rounds_correct_sensor_rate_list.append(correct_sensor_rate)
        rounds_undeter_sensor_rate_list.append(undeter_sensor_rate)
        rounds_incorrect_sensor_rate_list.append(incorrect_sensor_rate)
    
    
    desc_acc = [most_r, round(np.average(rounds_correct_sensor_rate_list), 4), round(np.average(rounds_incorrect_sensor_rate_list), 4), round(np.average(rounds_undeter_sensor_rate_list), 4)]
    df_desc_acc = df_desc_acc.append(pd.DataFrame([desc_acc], columns=['sensor_rate', 'correct_rate', 'incorrect_rate', 'undeter_rate']))

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
plt.xlabel("Sensor Rate, r", fontsize=18)
plt.ylabel("Active Sensor Proportion", fontsize=18)
plt.tick_params(labelsize=8)
plt.grid(True)