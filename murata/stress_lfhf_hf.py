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
import datetime
import matplotlib.dates as mdates
from tqdm import tqdm
import cv2
import scipy.stats as sp
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

output_folder = 'output/'
sj_list =['NS', 'FK', 'MM', '益田', 'F4', 'YM']
sj_r_list =['Sakai', 'Kawashima', 'Minowa', 'F.Masuda', 'Takahashi', 'M.Masuda']
sj_name = sj_list[5]

lfhf_ave_thd = 3
lfhf_std_thd = 2.4
hf_ave_thd = 640
hf_std_thd = 300


raw_data_pass = glob.glob(output_folder +sj_name + '_raw_data.csv')[0]
#sj_name = raw_data_pass.split('_')[0].split('\\')[1]
raw_data = pd.read_csv(raw_data_pass, engine='python', encoding="SHIFT-JIS")


numeric_columns = list(raw_data.columns)
del numeric_columns[:8]
none_numeric_columns = ['time', 'HR', 'LF', 'HF', 'LF_HF', 'LF+HF', 'ccvTP', 'judgement']
raw_data.columns = none_numeric_columns + numeric_columns
raw_data['time'] = pd.to_datetime(raw_data['time'])


if sj_name == sj_list[0]:
    sj_name = sj_r_list[0]
elif sj_name == sj_list[1]:
    sj_name = sj_r_list[1]
elif sj_name == sj_list[2]:
    sj_name = sj_r_list[2]
elif sj_name == sj_list[3]:
    sj_name = sj_r_list[3]

    new_numeric_rows = pd.DataFrame()
    for clm in numeric_columns:
        raw_data[clm] = raw_data[clm].where(raw_data[clm].str.contains('度'))
        split_row = raw_data[clm].str.split(' ', expand=True)
        for sub_clm in split_row.columns:
            split_row[sub_clm] = split_row[sub_clm].astype(str).str.strip('度')
        new_numeric_rows = pd.concat([new_numeric_rows, split_row], axis = 1)
    
    new_numeric_rows.columns = [i for i in range(0, len(new_numeric_rows.columns))]
    raw_data = raw_data.drop(numeric_columns,  axis = 1)
    raw_data = pd.concat([raw_data, new_numeric_rows], axis = 1)
    numeric_columns = new_numeric_rows.columns

elif sj_name == sj_list[4]:
    sj_name = sj_r_list[4]
elif sj_name == sj_list[5]:
    sj_name = sj_r_list[5]

for clm in numeric_columns:
    raw_data[clm] = pd.to_numeric(raw_data[clm], errors = 'coerce')


average_data = pd.DataFrame(columns = raw_data.columns)
start_index = 0
for index, line in raw_data.iterrows():
    if (line['time'] - raw_data.iloc[start_index]['time']) > datetime.timedelta(seconds=3600):
        ave_row = raw_data.iloc[start_index:index][raw_data.columns.drop('time')].mean()
        ave_row['time'] = raw_data.iloc[index - 1]['time']
        average_data = average_data.append(ave_row, ignore_index = True)
        start_index = index
        
    if index == len(raw_data) - 1:
        ave_row = raw_data.iloc[start_index:index + 1][raw_data.columns.drop('time')].mean()
        ave_row['time'] = raw_data.iloc[index]['time']
        average_data = average_data.append(ave_row, ignore_index = True)
 
average_data['time'] = pd.to_datetime(average_data['time'])
average_data.to_csv(output_folder + sj_name +'_average_data.csv')

trim_data = pd.DataFrame(columns = average_data.columns)
trim_data = average_data.dropna(subset = numeric_columns, how = 'all')
del_columns = ['judgement'] + list(numeric_columns)
trim_data['temp'] = trim_data[numeric_columns].mean(axis = 1)
#trim_data.index = trim_data['time']
trim_data = trim_data.drop(del_columns, axis = 1)
trim_data.to_csv(output_folder + sj_name + "_trim_data.csv")
trim_data['diff_time'] = [raw_time - datetime.datetime(raw_time.year, raw_time.month, raw_time.day) for raw_time  in trim_data['time']]
trim_data = trim_data.reset_index(drop = True)

#df_grouped = trim_data.set_index([trim_data.index.date, trim_data.index.weekday, trim_data.index])
#df_grouped.index.names = ['date', 'weekday', 'time']

label_time = pd.Series(trim_data['time'])
for index, df in trim_data.iterrows():
    if df['diff_time'] < datetime.timedelta(seconds=3600 * 3):
        label_time.iloc[index] = label_time.iloc[index] - df['diff_time'] - datetime.timedelta(days=1) 
    else:
        label_time.iloc[index] = label_time.iloc[index] - df['diff_time']
        
trim_data['label_time'] = label_time
trim_data.index = trim_data['label_time']
trim_data.index.name = 'label_time'
del_columns = ['time', 'diff_time', 'label_time']
trim_data = trim_data.drop(del_columns, axis = 1)

#rem_indexs = [group[0] for group in trim_data.groupby(level = 0) if len(group[1]) == 1]
#trim_data = trim_data.drop(rem_indexs)

df_ave = trim_data.mean(level = 'label_time').fillna(0)
df_std = trim_data.std(level = 'label_time').fillna(0)

df_ave_norm = pd.DataFrame(columns = df_ave.columns, index = df_ave.index)
df_std_norm = pd.DataFrame(columns = df_std.columns, index = df_std.index)

for clm in trim_data:
    df_ave_norm[clm] = sp.stats.zscore(df_ave[clm], axis=0)
    df_std_norm[clm] = sp.stats.zscore(df_std[clm], axis=0)

df_ave['month_day'] = [ str(raw_time.month) + '/' + str(raw_time.day)  for raw_time in df_ave.index]
df_std['month_day'] = [ str(raw_time.month) + '/' + str(raw_time.day)  for raw_time in df_std.index]
df_ave.index = df_ave['month_day']
df_std.index = df_std['month_day']

df_ave_norm['month_day'] = [ str(raw_time.month) + '/' + str(raw_time.day)  for raw_time in df_ave_norm.index]
df_std_norm['month_day'] = [ str(raw_time.month) + '/' + str(raw_time.day)  for raw_time in df_std_norm.index]
df_ave_norm.index = df_ave_norm['month_day']
df_std_norm.index = df_std_norm['month_day']


for data_kind in ['normal', 'norm']:
    for eva_kind in ['ave', 'std']:
        df_data = pd.DataFrame()
        
        if data_kind == 'normal':
            if eva_kind == 'ave':
                df_data = df_ave
            elif eva_kind == 'std':
                df_data = df_std
        elif data_kind == 'norm':
            if eva_kind == 'ave':
                df_data = df_ave_norm
            elif eva_kind == 'std':
                df_data = df_std_norm
        
        fig = plt.figure() 
        ax = fig.add_subplot(111) 
        ax2 = ax.twinx() 
        
        width = 0.3
        ax1 = df_data['LF_HF'].plot(kind='bar', color='b', ax=ax, width=width, position = 1)
        ax2 = df_data['HF'].plot(kind='bar', color='orange', ax=ax2, width=width, position = 0)
        
        ax1_y_max = round(ax1.get_ybound()[1] + 0.5)
        ax1_y_min = round(ax1.get_ybound()[0])
        ax2_y_max = round(ax2.get_ybound()[1] + 0.5)
        ax2_y_min = round(ax2.get_ybound()[0])
        
        most_ax1_y_max = max(abs(ax1_y_max), abs(ax1_y_min))
        most_ax2_y_max = max(abs(ax2_y_max), abs(ax2_y_min))
        
        if data_kind == 'norm':
            ax1.set_yticks(np.linspace(-1 *most_ax1_y_max, most_ax1_y_max, most_ax1_y_max * 2 + 1)) 
            ax2.set_yticks(np.linspace(-1 * most_ax2_y_max, most_ax2_y_max, most_ax2_y_max * 2 + 1)) 
            ax1.set_ylim([-1 * most_ax1_y_max, most_ax1_y_max])
            ax2.set_ylim([-1 * most_ax2_y_max, most_ax2_y_max])
        elif data_kind == 'normal':
            if eva_kind == 'ave':
                ax1.set_yticks(np.linspace(0, lfhf_ave_thd, 6)) 
                ax2.set_yticks(np.linspace(0, hf_ave_thd, 6)) 
                ax1.set_ylim([0, lfhf_ave_thd])
                ax2.set_ylim([0, hf_ave_thd])
            elif eva_kind == 'std':
                ax1.set_yticks(np.linspace(0, lfhf_std_thd, 6)) 
                ax2.set_yticks(np.linspace(0, hf_std_thd, 6)) 
                ax1.set_ylim([0, lfhf_std_thd])
                ax2.set_ylim([0, hf_std_thd])
                    
            
        
        ax1.set_title(sj_name + '_' + data_kind + '_' + eva_kind + '_LFHF_and_HF' , fontsize = 15)
        ax1.legend(loc='upper left', fontsize = 15)
        ax2.legend(loc='upper right', fontsize = 15)
        ax1.set_xlabel("time", fontsize = 15)
        
        ax.set_ylabel('LF/HF', fontsize = 15)
        ax2.set_ylabel('HF', fontsize = 15)
        ax1.tick_params(labelsize=15)
        ax2.tick_params(labelsize=15)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        
        ax2.figure.set_figwidth(16)
        ax2.figure.set_figheight(5)
        
        output_image_path = output_folder + sj_name + '_stress_lfhf_hf' + '_' + data_kind + '_' + eva_kind + '.png'
        plt.savefig(output_image_path)
        img = cv2.imread(output_image_path)
        plt.show()
        plt.close('all')

