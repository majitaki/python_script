import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import warnings
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import math
warnings.filterwarnings('ignore')

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

dir_list = glob.glob('*')
colorlist = ["g", "r", "grey"]


dim = ''
network_list=[]
max_weights = []
algo = ''
cur_round = 0

for mydir in dir_list:
    if os.path.isfile(mydir):
        continue
    cur_round = int(mydir.split('_')[7])
    network = mydir.split('_')[0]
    network_list.append(network) 
    algo = mydir.split('_')[5]
    commonweight = mydir.split('_')[9]
    commonweight = round(float(commonweight),2)
    max_weights.append(commonweight)
    
    
network_list=list(set(network_list))
max_weights=list(set(max_weights))
max_weights.sort()

df_monopoly = pd.DataFrame(columns = ['common_weight', 'network', 'value'])
df_minor = pd.DataFrame(columns = ['common_weight', 'network', 'value'])
df_undeter = pd.DataFrame(columns = ['common_weight', 'network', 'value'])
df_balance = pd.DataFrame(columns = ['common_weight', 'network', 'value'])
df_form_simpsond = pd.DataFrame(columns = ['common_weight', 'network', 'value'])


for network in network_list:
    accs = []
    none_accs = []
    undeter_accs = []
    cor_incor_accs = []
    better_simpsond = []
    std_better_simpsonds = []
    
    dir_network_list = glob.glob(network + '*')
    for mydir in tqdm(dir_network_list):
        if os.path.isfile(mydir):
            continue
        
        network = mydir.split('_')[0]
        dim = mydir.split('_')[3]
        cw = mydir.split('_')[9]
        os.chdir(mydir)
        csv_list = glob.glob('*.csv')
        df_all_ave = pd.DataFrame(columns=['monopoly_rate', 'minor_rate', 'undeter_rate', 'balance', 'form_simpsond'])
        
        for csv in csv_list:
            tmp = pd.read_csv(csv)
            tmp = tmp[['Round','CorrectRate', 'IncorrectRate', 'UndeterRate', 'BetterSimpsonD']]
            ave_green = round(tmp['CorrectRate'][(cur_round - 100):].mean() * 100,2)
            ave_red = round(tmp['IncorrectRate'][(cur_round - 100):].mean() * 100,2)
            ave_undeter = round(tmp['UndeterRate'][(cur_round - 100):].mean() * 100,2)
            ave_better_simpsond = round(tmp['BetterSimpsonD'][(cur_round - 100):].mean(),2)
            cor_incors = (tmp['CorrectRate'] + 0.0001) / (tmp['IncorrectRate'] + 0.0001)
            cor_incors = [math.log(data, 10) for data in cor_incors]
            df_ave_corincors = pd.DataFrame(cor_incors)
            ave_cor_incors = round(df_ave_corincors[(cur_round - 100):].std(), 2)[0]
            df_ave = pd.DataFrame([[ave_green, ave_red, ave_undeter, ave_cor_incors, ave_better_simpsond]], columns=['monopoly_rate', 'minor_rate', 'undeter_rate',  'balance', 'form_simpsond'])
            df_all_ave = df_all_ave.append(df_ave)
            
        for seed_value in df_all_ave['monopoly_rate']:
            df_monopoly = df_monopoly.append(pd.Series([cw, network, seed_value], index = ['common_weight', 'network', 'value']), ignore_index = True)
            
        for seed_value in df_all_ave['minor_rate']:
            df_minor = df_minor.append(pd.Series([cw, network, seed_value], index = ['common_weight', 'network', 'value']), ignore_index = True)
            
        for seed_value in df_all_ave['undeter_rate']:
            df_undeter = df_undeter.append(pd.Series([cw, network, seed_value], index = ['common_weight', 'network', 'value']), ignore_index = True)
            
        for seed_value in df_all_ave['balance']:
            df_balance = df_balance.append(pd.Series([cw, network, seed_value], index = ['common_weight', 'network', 'value']), ignore_index = True)

        for seed_value in df_all_ave['form_simpsond']:
            df_form_simpsond = df_form_simpsond.append(pd.Series([cw, network, seed_value], index = ['common_weight', 'network', 'value']), ignore_index = True)

        os.chdir('../')

ax = sns.lineplot(x = "common_weight" ,y="value", data=df_monopoly, marker = 'o', hue = 'network')
#ax = sns.lineplot(x = "common_weight" ,y="value", data=df_undeter, marker = 'o', label = 'UR')
#ax.set_title('monopoly_cw_' + algo + '_dim' + str(dim))
ax.set_ylabel("Dominant Rate(%)", fontsize=18)
ax.set_xlabel("common weight, cw", fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_yticks(np.arange(0, 105, 10))
ax.tick_params(labelsize = 20)
ax.figure.set_figwidth(10)
ax.figure.set_figheight(4)
ax.legend(fontsize=15)
plt.savefig('cw_monopoly_' +  str(dim) + '.png')
plt.show()
plt.close('all')

ax = sns.lineplot(x = "common_weight" ,y="value", data=df_minor, marker = 'o', hue = 'network')
#ax.set_title('minor_cw_' + algo + '_dim' + str(dim))
ax.set_ylabel("Minor Rate(%)", fontsize=18)
ax.set_xlabel("Common Weight, CW", fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_yticks(np.arange(0, 105, 10))
ax.tick_params(labelsize = 20)
ax.figure.set_figwidth(10)
ax.figure.set_figheight(4)
ax.legend(fontsize=20)
plt.savefig('cw_minor_' +  str(dim) + '.png')
plt.show()
plt.close('all')

ax = sns.lineplot(x = "common_weight" ,y="value", data=df_undeter, marker = 'o', hue = 'network')
#ax.set_title('undeter_cw_' + algo + '_dim' + str(dim))
ax.set_ylabel("Undeter Rate(%)", fontsize=18)
ax.set_xlabel("Common Weight, CW", fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_yticks(np.arange(0, 105, 10))
ax.tick_params(labelsize = 20)
ax.figure.set_figwidth(10)
ax.figure.set_figheight(4)
ax.legend(fontsize=20)
plt.savefig('cw_undeter_' +  str(dim) + '.png')
plt.show()
plt.close('all')

ax = sns.lineplot(x = "common_weight" ,y="value", data=df_balance, marker = 'o', hue = 'network')
#ax.set_title('balance_cw_' + algo + '_dim' + str(dim))
ax.set_ylabel("Balance", fontsize=18)
ax.set_xlabel("Common Weight, CW", fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_yticks(np.arange(0.0, 3.0, 0.5))
ax.tick_params(labelsize = 20)
ax.figure.set_figwidth(10)
ax.figure.set_figheight(4)
ax.legend(fontsize=20)
plt.savefig('cw_balance_' +  str(dim) + '.png')
plt.show()
plt.close('all')

ax = sns.lineplot(x = "common_weight" ,y="value", data=df_form_simpsond, marker = 'o', hue = 'network')
#ax.set_title('form_simpson_cw_' + algo + '_dim' + str(dim))
ax.set_ylabel("Form D, FD", fontsize=18)
ax.set_xlabel("Common Weight, CW", fontsize=18)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.tick_params(labelsize = 20)
ax.figure.set_figwidth(10)
ax.figure.set_figheight(4)
ax.legend(fontsize=20)
plt.savefig('cw_form_simpsond_' +  str(dim) + '.png')
plt.show()
plt.close('all')
