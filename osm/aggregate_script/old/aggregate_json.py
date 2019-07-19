import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

dir_list = glob.glob('*')
colorlist = ["g", "r", "grey"]

cur_round = 0

for mydir in dir_list:
    if os.path.isfile(mydir):
        continue
    cur_round = int(mydir.split('_')[7])

for dir in tqdm(dir_list):
    if os.path.isfile(dir):
        continue
    os.chdir(dir)
    csv_list = glob.glob('*.csv')
    df_all_ave = pd.DataFrame(columns=['correct_rate', 'incorrect_rate', 'undeter_rate'])
    df_all_not_good = pd.DataFrame()
    
    
    
    for csv in csv_list:
        tmp = pd.read_csv(csv)
        tmp = tmp[['Round','CorrectRate', 'IncorrectRate', 'UndeterRate']]
        ave_green = round(tmp['CorrectRate'].mean(),2)
        ave_red = round(tmp['IncorrectRate'].mean(),2)
        ave_undeter = round(tmp['UndeterRate'].mean(),2)
        #not_good_rate = tmp['correct_rate'].describe()['std']
        #not_good_rate = sum(tmp['green_rate'][100:] < 0.5) / tmp['green_rate'].count()
        df_ave = pd.DataFrame([[ave_green, ave_red, ave_undeter]], columns=['correct_rate', 'incorrect_rate', 'undeter_rate'])
        df_all_ave = df_all_ave.append(df_ave)
        #df_not_good = pd.DataFrame([[not_good_rate]])
        #df_all_not_good = df_all_not_good.append(df_not_good)
        plt.figure()
        ax = tmp.plot(title = csv, 
                      kind = 'area', 
                      x = tmp.columns[0], 
                      color = colorlist, 
                      alpha=0.5,
                      figsize=(20,5)
                      )
        ax.legend([tmp.columns[1] + ' ' + str(ave_green), tmp.columns[2] + ' ' + str(ave_red), tmp.columns[3] + ' ' + str(ave_undeter)])
        plt.savefig(csv + '.png')
        plt.close('all')
    os.chdir('../')
    all_ave_green = round(df_all_ave['correct_rate'].mean(),3)
    all_ave_red = round(df_all_ave['incorrect_rate'].mean(),3)
    all_ave_undeter = round(df_all_ave['undeter_rate'].mean(),3)
    #all_not_good_rate = round(df_all_not_good[0].mean(),3)
    file_name = './' + dir
    df_all_ave.to_csv( file_name + ".csv")