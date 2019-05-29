import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import seaborn as sns
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

result_list = glob.glob('*.csv')
colorlist = ["g", "r", "grey"]

network_list=[]
algo_list=[]
node_list=[]
dim_list=[]

for result in result_list:
    network = result.split('_')[0]
    node = result.split('_')[1]
    sensor_mode = result.split('_')[2]
    dim_size = result.split('_')[3]
    sensor_rate = result.split('_')[4]
    algo = result.split('_')[5]
    
    network_list.append(network)
    algo_list.append(algo)
    node_list.append(int(node))
    dim_list.append(int(dim_size))

network_list=list(set(network_list))
algo_list=list(set(algo_list))
node_list=list(set(node_list))
node_list.sort()
dim_list = list(set(dim_list))
dim_list.sort()
kind_list = ['cor', 'incor', 'undeter']

for dim in dim_list:
    for network in tqdm(network_list):
        for kind in kind_list:
            plt.figure(figsize=(15,8))
            for algo in algo_list:
                df_t_list = []
                data_list = glob.glob('*' + network +'*'+'_' +str(dim)+'_' + '*' + algo +'_'+ '*' + '.csv')
                df_green = pd.DataFrame(columns = node_list)
                df_red = pd.DataFrame(columns = node_list)
                df_undeter = pd.DataFrame(columns = node_list)
                for s_data in data_list:
                    for node in node_list:
                        match = re.search(str(node), s_data)
                        if match:
                            df_green_csv = pd.read_csv(s_data)['correct_rate']
                            df_red_csv = pd.read_csv(s_data)['incorrect_rate']
                            df_undeter_csv = pd.read_csv(s_data)['undeter_rate']
                            df_green[node] = df_green_csv
                            df_red[node] = df_red_csv
                            df_undeter[node] = df_undeter_csv
                df_green_t = df_green.T
                df_red_t = df_red.T
                df_undeter_t = df_undeter.T
                num_columns = len(df_green_t.columns)
                df_green_t_list = [df_green_t[i] for i in range(num_columns)]
                df_red_t_list = [df_red_t[i] for i in range(num_columns)]
                df_undeter_t_list = [df_undeter_t[i] for i in range(num_columns)]
                
                my_color = '';
                if algo == 'AATG':
                    my_color = 'green'
                elif algo == 'AAT':
                    my_color = 'gray'
                elif algo == 'AATfix':
                    my_color = 'blue'
                elif algo == 'OSMonly':
                    my_color = 'black'
                elif algo == 'IWTori':
                    my_color = 'Magenta'
                    
                    
                data = []
                if kind == 'cor':
                    data = df_green_t_list
                elif kind == 'incor':
                    data = df_red_t_list
                elif kind == 'undeter':
                    data = df_undeter_t_list
                    
                sns.tsplot(data, time = node_list, marker = 'o', color = my_color, ci = 68)
                
            label_list = algo_list
    #        if algo_list[0] == 'AATG':
    #            label_list = ['AATG' , 'AAT']
    #        else:
    #            label_list = ['AAT' , 'AATG']
                
                
            #plt.ylim([0, 1.0])
            #plt.xticks(np.arange(0, 1.0, 0.1))
            plt.title( network + '_' + str(kind))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xlabel("Network size, N", fontsize=18)
            plt.ylabel("Accuracy, R", fontsize=18)
            plt.tick_params(labelsize=20)
            plt.grid(True)
            plt.legend(labels=label_list, fontsize=15)
            plt.savefig(str(kind) + '_' + network + '_' +  str(dim) + '.png')
            plt.close('all')