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
my_belief_setting = osm.Belief_Setting.BayesFilter
#my_belief_setting = osm.Belief_Setting.ParticleFilter
my_samples = 5

dim = 10
#step_size = 1500
step_duration = 500
rounds = 3
threshold = 0.90
sensor_size = 30
op_intro_rate = 0.1
op_intro_duration = 10
op_intro_size = 3 * sensor_size
duration_sensor_rate = 0.05
my_sensor_acc = 0.8


average_columns = ['correct_rate', 'incorrect_rate', 'undeter_rate']
graph_columns = ['sensor_acc'] + average_columns
graph_three_columns = ['sensor_acc', 'value', 'rate_kind']
rcv_columns = ['sensor_acc'] + ['receive_num']

df_desc_acc = pd.DataFrame(columns = graph_columns)
df_desc_all = pd.DataFrame(columns = graph_three_columns)
df_desc_rcv_num = pd.DataFrame(columns = rcv_columns)

env_array = None
thete_map = None

my_sensor_weight = osm.make_sensor_acc(my_agent_weight_setting, my_sensor_acc, my_fix_sensor_weight)
pre_p_array = np.full(dim, 1.0 / dim)
df_opinion_by_steps = pd.DataFrame(columns = ['op_value', 'is_correct', 'correct_op'])


for t_index in range(dim):
    env_array = osm.make_env(my_env_dist, dim, False, sensor_acc = my_fix_turara, turara_index = t_index)
    thete_map = env_array.argmax()
    results = osm.sensor_simulation_by_step(step_duration, 
                                            threshold,
                                            op_intro_rate,
                                            op_intro_duration,
                                            sensor_size,
                                            env_array,
                                            my_sensor_weight,
                                            my_belief_setting,
                                            my_samples, pre_p_array)
    pre_p_array = results[1]
    df_each = pd.DataFrame(columns = ['op_value', 'is_correct', 'correct_op'])
    df_each['op_value'] = results[0]
    df_each['is_correct'] = df_each['op_value'] == thete_map
    df_each['correct_op'] = thete_map
    df_opinion_by_steps = df_opinion_by_steps.append(df_each)

df_opinion_by_steps = df_opinion_by_steps.reset_index(drop = True)
correctness = df_opinion_by_steps.is_correct.sum() / len(df_opinion_by_steps.is_correct)

plt.title("Opinion Accuracy " + str(correctness))
df_opinion_by_steps.is_correct.astype(int).plot()
plt.show()
plt.close('all')
plt.title("Opinion Changeness by steps")
df_opinion_by_steps.op_value.plot()
df_opinion_by_steps.correct_op.astype(int).plot()











