import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
np.set_printoptions(precision=3)
import seaborn as sns
import osm_functions as osm


my_env_dist = osm.Env_Dist.Turara_Fix
my_fix_turara = 0.6
my_agent_weight_setting = osm.Agent_Weight_Setting.Sensor_Weight_Depend_Sensor_Acc
my_fix_sensor_weight = 0.55
#my_belief_setting = osm.Belief_Setting.BayesFilter
my_belief_setting = osm.Belief_Setting.ParticleFilter
my_samples = 5

dim = 2
step_size = 1500
rounds = 3
threshold = 0.90
sensor_size = 30
op_intro_rate = 0.1
op_intro_duration = 10
op_intro_size = 3 * sensor_size
duration_sensor_rate = 0.05

average_columns = ['correct_rate', 'incorrect_rate', 'undeter_rate']
graph_columns = ['sensor_acc'] + average_columns
graph_three_columns = ['sensor_acc', 'value', 'rate_kind']
rcv_columns = ['sensor_acc'] + ['receive_num']

df_desc_acc = pd.DataFrame(columns = graph_columns)
df_desc_all = pd.DataFrame(columns = graph_three_columns)
df_desc_rcv_num = pd.DataFrame(columns = rcv_columns)

env_array = None
thete_map = None

if my_env_dist != osm.Env_Dist.Turara_Depend_Sensor_Acc:
    env_array = osm.make_env(my_env_dist, dim, False, sensor_acc = my_fix_turara, turara_index = 0)
    thete_map = env_array.argmax()

sensor_accs = None
sensor_accs = np.arange(0, 1.0 + duration_sensor_rate, duration_sensor_rate)


for sensor_acc in sensor_accs:
   rounds_correct_sensor_rate_list =[]
   rounds_undeter_sensor_rate_list =[]
   rounds_incorrect_sensor_rate_list =[]
   rounds_receive_num_rate_list = []

   for r in tqdm(range(rounds), desc = "Sensor Acc: " + str(round(sensor_acc, 4))):
       df_output_op = pd.DataFrame()
       df_receive_num = pd.DataFrame()

       if my_env_dist == osm.Env_Dist.Turara_Depend_Sensor_Acc:
           env_array = osm.make_env(my_env_dist, dim, True, sensor_acc = sensor_acc, turara_index = 0)
           thete_map = env_array.argmax()
           
       my_sensor_weight = osm.make_sensor_acc(my_agent_weight_setting, sensor_acc, my_fix_sensor_weight)

       results = Parallel(n_jobs = 1)(
               [delayed(osm.sensor_simulation)(step_size, threshold, op_intro_rate, op_intro_duration, sensor_size, env_array, my_sensor_weight, my_belief_setting, my_samples)
               for j in range(sensor_size)]
               )
       for result in results:
           df_output_op = df_output_op.append(pd.DataFrame(result[0]).T)
           df_receive_num = df_receive_num.append([result[2]])
       
       df_output_op = df_output_op.reset_index(drop=True)
       df_receive_num = df_receive_num.reset_index(drop=True)
        
       df_correct = pd.DataFrame(df_output_op[thete_map] == 1)
       df_undeter = pd.DataFrame(df_output_op.sum(axis = 1))
       correct_sensor_rate = df_correct.where(df_correct == True).dropna().count() / sensor_size
       undeter_sensor_rate = df_undeter.where(df_undeter == 0).dropna().count()[0] / sensor_size
       incorrect_sensor_rate = 1.0 - correct_sensor_rate - undeter_sensor_rate

       rounds_correct_sensor_rate_list.append(correct_sensor_rate.values[0])
       rounds_undeter_sensor_rate_list.append(undeter_sensor_rate)
       rounds_incorrect_sensor_rate_list.append(incorrect_sensor_rate.values[0])
       rounds_receive_num_rate_list.append(df_receive_num.mean()[0])
   
   desc_acc = [sensor_acc, 
               round(np.average(rounds_correct_sensor_rate_list), 4), 
               round(np.average(rounds_incorrect_sensor_rate_list), 4), 
               round(np.average(rounds_undeter_sensor_rate_list), 4),
               ]
   
   df_each_correct = pd.DataFrame([[sensor_acc for i in range(rounds)],
                                    rounds_correct_sensor_rate_list,
                                    [average_columns[0] for i in range(rounds)]]).T
   df_each_incorrect = pd.DataFrame([[sensor_acc for i in range(rounds)],
                                    rounds_incorrect_sensor_rate_list,
                                    [average_columns[1] for i in range(rounds)]]).T
   df_each_undeter = pd.DataFrame([[sensor_acc for i in range(rounds)],
                                    rounds_undeter_sensor_rate_list,
                                    [average_columns[2] for i in range(rounds)]]).T
   
   df_desc_acc = df_desc_acc.append(pd.DataFrame([desc_acc], columns = graph_columns))
   
   df_each_desc_all = pd.DataFrame().append(df_each_correct).append(df_each_incorrect).append(df_each_undeter)
   df_each_desc_all.columns = graph_three_columns
   df_desc_all = df_desc_all.append(df_each_desc_all)
   df_desc_all["sensor_acc"] = df_desc_all["sensor_acc"].astype(float)
   df_desc_all["value"] = df_desc_all["value"].astype(float)
   
   df_each_desc_rcv = pd.DataFrame([[sensor_acc for i in range(rounds)],
                                    rounds_receive_num_rate_list,
                                    ]).T
   df_each_desc_rcv.columns = rcv_columns
   df_desc_rcv_num = df_desc_rcv_num.append(df_each_desc_rcv)
   

plt.figure()
colorlist = ["g", "r", "grey"]
ax = df_desc_acc.plot(title = 'DimSize:' + str(dim) + ' StepSize:' + str(step_size),
             kind = 'area',
             x = df_desc_acc.columns[0],
             y = average_columns,
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
plt.show()
plt.close('all')

plt.figure(figsize=(8,5))
sns.lineplot(x = "sensor_acc", y = "value", hue = "rate_kind", data = df_desc_all, palette = colorlist)
plt.ylim([0, 1.0])
plt.xlim([0, 1.0])
plt.xticks(np.arange(0.0, 1.0, 0.05))
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.xlabel("Sensor Acc", fontsize=18)
plt.ylabel("Sensor Opinion Acc", fontsize=18)
plt.tick_params(labelsize=8)
plt.grid(True)
plt.show()
plt.close('all')


plt.figure(figsize=(8,5))
sns.lineplot(x = "sensor_acc", y = "receive_num", data = df_desc_rcv_num)
plt.xlim([0, 1.0])
plt.xticks(np.arange(0.0, 1.0, 0.05))
plt.xlabel("Sensor Acc", fontsize=18)
plt.ylabel("Sensor Receive Num", fontsize=18)
plt.tick_params(labelsize=8)
plt.grid(True)
plt.show()
plt.close('all')