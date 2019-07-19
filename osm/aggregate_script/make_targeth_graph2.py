import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import warnings
import numpy as np
import seaborn as sns
import math
warnings.filterwarnings('ignore')

dir_list = glob.glob('*')
colorlist = ["g", "r", "grey"]

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

dim = ''
network_size = ""
network_list=[]
max_targets = []
algo = ''
cur_round = 0

for mydir in dir_list:
    if os.path.isfile(mydir):
        continue
    cur_round = int(mydir.split('_')[7])
    network = mydir.split('_')[0]
    network_list.append(network) 
    algo = mydir.split('_')[5]
    targeth = mydir.split('_')[9]
    targeth = round(float(targeth),2)
    max_targets.append(targeth)
    
    
network_list=list(set(network_list))
max_targets=list(set(max_targets))
max_targets.sort()

df_target = pd.DataFrame(index = max_targets, columns=network_list)
df_incorrect_target = pd.DataFrame(index = max_targets, columns=network_list)
df_undeter_target = pd.DataFrame(index = max_targets, columns=network_list)
df_cor_incor_target = pd.DataFrame(index = max_targets, columns=network_list)

for network in network_list:
    accs = []
    none_accs = []
    undeter_accs = []
    cor_incor_accs = []
    
    dir_network_list = glob.glob(network + '*')
    for mydir in tqdm(dir_network_list):
        if os.path.isfile(mydir):
            continue
        
        network = mydir.split('_')[0]
        network_size = mydir.split('_')[1]
        dim = mydir.split('_')[3]
        os.chdir(mydir)
        csv_list = glob.glob('*.csv')
        df_all_ave = pd.DataFrame(columns=['correct_rate', 'incorrect_rate', 'undeter_rate', 'cor/incor'])
        
        for csv in csv_list:
            tmp = pd.read_csv(csv)
            tmp = tmp[['Round','CorrectRate', 'IncorrectRate', 'UndeterRate']]
            ave_green = round(tmp['CorrectRate'].mean(),2)
            ave_red = round(tmp['IncorrectRate'].mean(),2)
            ave_undeter = round(tmp['UndeterRate'].mean(),2)
            cor_incors =  (tmp['CorrectRate']) /  ((tmp['IncorrectRate'] + tmp['UndeterRate']) + 1)
            #cor_incors = [math.log2(data + 1) for data in cor_incors]
            df_ave_corincors = pd.DataFrame(cor_incors)
            ave_cor_incors = round(df_ave_corincors.mean(),2)
            df_ave = pd.DataFrame([[ave_green, ave_red, ave_undeter, ave_cor_incors]], columns=['correct_rate', 'incorrect_rate', 'undeter_rate',  'cor/incor'])
            df_all_ave = df_all_ave.append(df_ave)
            
        each_correct_ave = df_all_ave['correct_rate'].mean()
        each_incorrect_ave = df_all_ave['incorrect_rate'].mean()
        each_undeter_ave = df_all_ave['undeter_rate'].mean()
        each_cor_incor_ave = df_all_ave['cor/incor'].mean()
        
        accs.append(round(each_correct_ave,3))
        none_accs.append(round(each_incorrect_ave,3))
        undeter_accs.append(round(each_undeter_ave,3))
        cor_incor_accs.append(round(each_cor_incor_ave,3))
        os.chdir('../')
    
    df_target[network] = pd.DataFrame(accs, index = max_targets[:len(accs)], columns = [network])
    df_incorrect_target[network] = pd.DataFrame(none_accs, index = max_targets[:len(none_accs)], columns = [network])
    df_undeter_target[network] = pd.DataFrame(undeter_accs, index = max_targets[:len(undeter_accs)], columns = [network])
    df_cor_incor_target[network] = pd.DataFrame(cor_incor_accs, index = max_targets[:len(none_accs)], columns = [network])
    
df_target.plot(grid = True, marker = 'o', figsize=(5,5))
plt.title('cor_targeth_' + algo + '_dim' + str(dim) + '_size' + str(network_size))
plt.legend(fontsize=20)
plt.xticks(np.arange(min(max_targets), max(max_targets) + 0.05, 0.2))
plt.yticks(np.arange(0.0, 1.05, 0.2))
plt.xlabel("Target awareness rate, htrg", fontsize=24)
plt.ylabel("Accuracy, R", fontsize=24)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('targeth_correct_' +  str(dim) + '.png')
#plt.close('all')

df_incorrect_target.plot(grid = True, marker = 'o', figsize=(5,5))
plt.title('incor_targeth_' + algo + '_dim' + str(dim) + '_size' + str(network_size))
plt.legend(fontsize=20)
plt.xticks(np.arange(min(max_targets), max(max_targets) + 0.05, 0.2))
plt.yticks(np.arange(0.0, 1.05, 0.2))
plt.xlabel("Target awareness rate, htrg", fontsize=24)
plt.ylabel("Not Accuracy, Not R", fontsize=24)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('targeth_incorrect_' +  str(dim) + '.png')

df_undeter_target.plot(grid = True, marker = 'o', figsize=(5,5))
plt.title('undeter_targeth_' + algo + '_dim' + str(dim) + '_size' + str(network_size))
plt.legend(fontsize=20)
plt.xticks(np.arange(min(max_targets), max(max_targets) + 0.05, 0.2))
plt.yticks(np.arange(0.0, 1.05, 0.2))
plt.xlabel("Target awareness rate, htrg", fontsize=24)
plt.ylabel("Undeter", fontsize=24)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('targeth_undeter_' +  str(dim) + '.png')


df_cor_incor_target.plot(grid = True, marker = 'o', figsize=(5,5))
plt.title('balance_targeth_' + algo + '_dim' + str(dim) + '_size' + str(network_size))
plt.legend(fontsize=20)
plt.xticks(np.arange(min(max_targets), max(max_targets) + 0.05, 0.2))
plt.yticks(np.arange(0.0, 1.05, 0.2))
plt.xlabel("Target awareness rate, htrg", fontsize=24)
plt.ylabel("Balance", fontsize=24)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('targeth_balance_' +  str(dim) + '.png')