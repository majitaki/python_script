# coding: shift_jis
'''
Created on 2017/06/23

@author: gonbe
'''

import os
import glob
import pandas as pd
from tqdm import tqdm

file_path = os.getcwd()

KK_path = file_path + "\KK\*"
P_path = file_path + "\P\*"

KK_name = glob.glob(KK_path)
P_name = glob.glob(P_path)


def judge(lf_hf,ccvtp):
    CCVTP_THRESHOLD=2.859
    CCVTP_HIGH_THRESHOLD=7.659

    if(lf_hf>5):
        if(ccvtp>CCVTP_HIGH_THRESHOLD):
            return "2!!!"
        else:
            return 2
    elif(lf_hf>2):
        if(ccvtp>CCVTP_HIGH_THRESHOLD):
            return "1!!!"
        else:
            return 1
    elif(ccvtp>CCVTP_THRESHOLD):
        if(ccvtp>CCVTP_HIGH_THRESHOLD):
            return "0!!!"
        else:
            return 0
    else:
        if(ccvtp>CCVTP_HIGH_THRESHOLD):
            return "1!!!"
        else:
            return 1


all_KK_data = pd.DataFrame()
sj_name = ''
for name in tqdm(KK_name):
    if os.path.isdir(name):
        continue
    path, ext = os.path.splitext( os.path.basename(name) )
    
    sj_name = path.split("_")[2]
    date = path.split("_")[3]
    time = path.split("_")[4]

    year = date[:4]
    month = date[4:6]
    day = date[6:]
    hour = time[:2]
    min = time[2:4]
    sec = time[4:]

    date_time = year + "/" + month + "/" + day + " "  + hour + ":" + min + ":" + sec

    data = pd.read_csv(name, index_col=None, header=None, skiprows=1, engine='python', encoding="SHIFT-JIS")
    data.columns = ["RRI", "HR", "HF", "LF", "LF+HF", "LF/HF", "SD(LF/HF)", "CVRR", "ccvTP", "ln(CcvTP)"]
    data.index = [date_time]

    data = data[["HR", "LF", "HF", "LF/HF", "LF+HF", "ccvTP"]]

    data = pd.concat([data, pd.DataFrame(judge(lf_hf=data["LF/HF"].item(),ccvtp=data["ccvTP"].item()), index=data.index, columns =["judgement"])], axis=1)

    all_KK_data = pd.concat([all_KK_data, data], axis=0)

all_KK_data = all_KK_data.dropna()




all_P_data = pd.DataFrame()
for name in tqdm(P_name):
    if os.path.isdir(name):
        continue
    path, ext = os.path.splitext( os.path.basename(name) )

    date = path.split("_")[3]
    time = path.split("_")[4]

    year = date[:4]
    month = date[4:6]
    day = date[6:]
    hour = time[:2]
    min = time[2:4]
    sec = time[4:]

    date_time = year + "/" + month + "/" + day + " "  + hour + ":" + min + ":" + sec
    
    try:
        data = pd.read_csv(name, skiprows=1, engine='python', encoding="SHIFT-JIS" )
    except:
        data = pd.read_csv(name, skiprows=1, engine='python', encoding="SHIFT-JIS", sep='^')
        data_columns = pd.Series(data.columns[0])
        data_columns = data_columns.str.split(',', expand = True)
        data = data[data.columns[0]].str.split(',', expand = True)
        data = data.drop(columns = [clm for clm in range(len(data_columns.iloc[0]),  len(data.columns))])
        data.columns = data_columns.iloc[0]

    event = data["Event"].dropna()
    event = event.reset_index()
    event = event["Event"]
    event.name = date_time

    all_P_data = pd.concat([all_P_data, event.T], axis=1)

all_P_data = all_P_data.T


data_two = pd.concat([all_KK_data, all_P_data], axis=1)
data_two = data_two.dropna(subset=["HR"])

data_two.to_csv("output/" + sj_name + "_raw_data.csv", encoding="SHIFT-JIS")

