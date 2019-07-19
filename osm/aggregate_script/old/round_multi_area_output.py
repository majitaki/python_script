# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
import glob
from statistics import mean
import pandas.tools.plotting as plotting
import matplotlib.pyplot as plt
import cv2

algo = "aatti"
network = 'ws'
file_name = ("RoundOpinion_WS_gs1_node100_sensor5_AATti_as1_round200_step1000_0.85_*_001")
folder = ("")
csv_list = glob.glob("../" + folder + file_name + ".csv")

for csv in csv_list:
    data = pd.read_csv(csv)
    data["round kai"] =  data["round"]
    
    df = pd.DataFrame({
                    "round" : data["round kai"],
                    "White" : data["correct"],
                    "Black" : data["incorrect"],
                    "Undeter" : data["undeter"]
                    })
    
    
    #df[df["round"] >= 0].loc[:, ["correct","incorrect", "undeter"]].plot.area(alpha = 0.4, figsize=(10,3))
    ax = df[df["round"] <= 200][df["round"] >= 0].plot.area(
            fontsize = 20,
            #title = csv,
            x = ["round"],
            y = ["White", "Black", "Undeter"], 
            alpha = 0.9, 
            figsize=(8,5), 
            linewidth = 0.0
            )
    title = ''
    if 'Undeter' in csv:
        title = '(a) Undeter Initial State'
    elif 'White' in csv:
        title = '(b) Correct Initial State'
    elif 'Black' in csv:
        title = '(c) Incorrect Initial State'
    
    acc = df["White"].mean()
    plt.title(title, fontsize = 20)
    #plt.title(title +" (R: "+ str(acc) + ")")
    plt.xlabel('Round')
    if('Undeter' in csv):
        plt.ylabel('Accuracy R')
    else:
        ax.tick_params(labelleft="off",left="off") # y軸の削除
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    plt.legend(prop={'size':20})
    plt.tight_layout()
    #plt.savefig("./result/" + str(csv + ".png"))
    file_name = "area_"+ network +"_"+algo+ "_"+ str(title) + ".png"
    plt.savefig('./' + file_name, bbox_inches='tight')
    
    

img1u = cv2.imread('area_'+ network +'_' + algo + '_(a) Undeter Initial State.png')
img1w = cv2.imread('area_'+ network +'_' + algo + '_(b) Correct Initial State.png')
img1b = cv2.imread('area_'+ network +'_' + algo + '_(c) Incorrect Initial State.png')
img1 = cv2.hconcat([img1u, img1w, img1b])
#img2 = cv2.hconcat([img2u, img2w, img2b])
#img = cv2.vconcat([img1, img2])
cv2.imwrite('area_' + network + '_'+ algo +'.png', img1)


