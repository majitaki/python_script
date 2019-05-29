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
import statsmodels.api as sm

sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})

output_folder = 'output/'
sj_list =['NS', 'FK', 'MM', '益田', 'F4', 'YM']
sj_r_list =['Sakai', 'Kawashima', 'Minowa', 'F.Masuda', 'Takahashi', 'M.Masuda']
sj_name = sj_list[1]

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
del_columns = ['time', 'judgement'] + list(numeric_columns)
trim_data['temp'] = trim_data[numeric_columns].mean(axis = 1)
trim_data['LF_Norm'] = trim_data['LF'] / trim_data['LF+HF']
trim_data['HF_Norm'] = trim_data['HF'] / trim_data['LF+HF']

trim_data.index = trim_data['time']
trim_data = trim_data.drop(del_columns, axis = 1)
trim_data.to_csv(output_folder + sj_name + "_trim_data_norm.csv")

delta_days = (trim_data.index[-1] - trim_data.index[0]).days
res_freq = round(len(trim_data) / delta_days)

day_interval = 1
#graph_list = ['HR', 'LF', 'HF', 'LF_HF', 'LF+HF', 'ccvTP']
graph_list = ['temp', 'ccvTP', 'HR', 'LF', 'HF', 'LF_HF', 'LF+HF']
imgs = []

for index, clm in  tqdm(zip(range(0, len(graph_list)), graph_list)):
    res = sm.tsa.seasonal_decompose(trim_data[clm], freq = res_freq)
    
    ax1 = res.resid.plot(marker = 'o', label = clm, fontsize=15)
    
    ax1.set_xlabel("time", fontsize = 15)
    ax1.set_ylabel(clm, fontsize = 15)
    
    ax1.set_title(sj_name + '_' + clm , fontsize = 15)
    ax1.legend(loc='upper left', fontsize = 15)
    
    ax1_y_max = ax1.get_ybound()[1]
    ax1_y_min = ax1.get_ybound()[0]
    
    
    ax1.set_yticks(np.linspace(ax1_y_min, ax1_y_max, 6))
    ax1.set_ylim([ax1_y_min, ax1_y_max])
    
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval = day_interval))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.tick_params(axis="x", labelsize=15)
    
    ax1.figure.set_figwidth(16)
    ax1.figure.set_figheight(5)
    plt.setp( ax1.xaxis.get_majorticklabels(), rotation=0, ha="left", rotation_mode="anchor" ) 
    #plt.tight_layout()
    output_image_path = output_folder + sj_name + '_seasonal_' + clm +'.png'
    plt.savefig(output_image_path)
    img = cv2.imread(output_image_path)
    imgs.append(img)
    plt.show()
    plt.close('all')
    
united_imgs = cv2.vconcat(imgs)
cv2.imwrite(output_folder + 'output_compare_all_norm_residential_' + sj_name + '.png', united_imgs)