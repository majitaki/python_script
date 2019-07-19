import pandas as pd
import matplotlib.pyplot as plt
import glob
import pathlib
import os
import numpy as np
from tqdm import tqdm
import matplotlib.ticker as ticker
import warnings
import seaborn as sns; sns.set()
warnings.filterwarnings('ignore')

graph_set = set()
size_set = set()
algorithm_set = set()
th_set = set()
dim_set = set()

os.chdir('data')
p = pathlib.Path('./')
dir_list = [i for i in p.iterdir() if i.is_dir()]
for d in dir_list:
    graph_set.add(str(d).split('_')[0])
    size_set.add(int(str(d).split('_')[1]))
    algorithm_set.add(str(d).split('_')[2])
    dim_set.add(int(str(d).split('_')[3]))
    th_set.add(float(str(d).split('_')[4]))

df_th = pd.DataFrame(columns = ['graph', 'size', 'algo', 'dim', 'th', 'seed', 'round_correct'])

loop_num = len(graph_set) * len(size_set) * len(algorithm_set) * len(dim_set) * len(th_set)

with tqdm(total=loop_num) as pbar:
    for graph in graph_set:
        for size in size_set:
            for algo in algorithm_set:
                for dim in dim_set:
                    for th in th_set:
                        pbar.update(1)
                        target_path = p.glob(graph + '_' + str(size) + '_' + algo + '_' + str(dim) + '_' + str(th) + '_')
                        target_dir = list(target_path)
                        if len(target_dir) > 1:
                            warnings.warn("duplication error")
                        if len(target_dir) == 0:
                            continue
                        csv_list = list(target_dir[0].glob('*.csv'))
                        for csv in csv_list:
                            seed = str(csv).split('_')[-2]
                            df_csv = pd.read_csv(str(csv))
                            ave_green = round(df_csv['CorrectRate'].mean(),3)
                            df_th = df_th.append(pd.Series([graph,size,algo,dim,th,seed,ave_green], index = ['graph', 'size', 'algo', 'dim', 'th', 'seed', 'round_correct']), ignore_index = True)
                            
#graph
if len(graph_set) >= 1:
    loop_num = len(size_set) * len(algorithm_set) * len(dim_set)
    with tqdm(total=loop_num) as pbar:
        for size in size_set:
            for algo in algorithm_set:
                for dim in dim_set:
                    pbar.update(1)
                    tmp = df_th[df_th['size'] == size]
                    tmp = tmp[tmp['algo'] == algo]
                    tmp = tmp[tmp['dim'] == dim]
                    plt.figure()
                    ax = sns.lineplot(x="th",
                                      y="round_correct",
                                      style = 'graph',
                                      hue = 'graph',
                                      palette=sns.color_palette("Oranges_d", len(graph_set)),
                                      data=tmp,
                                      markers=True,
                                      )
                    plt.xticks(np.arange(0.0, 1.05, 0.2))
                    plt.title('th_' + 'size' + str(size) + '_algo' + str(algo) + '_dim' + str(dim))
                    plt.yticks(np.arange(0.0, 1.05, 0.2))
                    plt.legend(fontsize=20)
                    plt.xlabel("Target awareness rate, htrg", fontsize=24)
                    plt.ylabel("Accuracy, R", fontsize=24)
                    plt.tick_params(labelsize=20)
                    plt.tight_layout()
                    plt.savefig('../th_' + 'size' + str(size) + '_algo' + str(algo) + '_dim' + str(dim) + '.png')
                    

#size
if len(size_set) >= 1:
    loop_num = len(graph_set) * len(algorithm_set) * len(dim_set)
    with tqdm(total=loop_num) as pbar:
        for graph in graph_set:
            for algo in algorithm_set:
                for dim in dim_set:
                    pbar.update(1)
                    tmp = df_th[df_th['graph'] == graph]
                    tmp = tmp[tmp['algo'] == algo]
                    tmp = tmp[tmp['dim'] == dim]
                    plt.figure()
                    sns.set_palette("husl")
                    ax = sns.lineplot(x="th",
                                      y="round_correct",
                                      style = 'size',
                                      hue = 'size',
                                      palette=sns.color_palette("Blues_d", len(size_set)),
                                      data=tmp,
                                      markers=True
                                      )
                    plt.xticks(np.arange(0.0, 1.05, 0.2))
                    plt.title('th_' + 'graph' + str(graph) + '_algo' + str(algo) + '_dim' + str(dim))
                    plt.yticks(np.arange(0.0, 1.05, 0.2))
                    plt.legend(fontsize=20)
                    plt.xlabel("Target awareness rate, htrg", fontsize=24)
                    plt.ylabel("Accuracy, R", fontsize=24)
                    plt.tick_params(labelsize=20)
                    plt.tight_layout()
                    plt.savefig('../th_' + 'graph' + str(graph) + '_algo' + str(algo) + '_dim' + str(dim) + '.png')
                    
                
                
#dim
if len(dim_set) >= 1:
    loop_num = len(size_set) * len(algorithm_set) * len(graph_set)
    with tqdm(total=loop_num) as pbar:
        for graph in graph_set:
            for algo in algorithm_set:
                for size in size_set:
                    pbar.update(1)
                    tmp = df_th[df_th['graph'] == graph]
                    tmp = tmp[tmp['algo'] == algo]
                    tmp = tmp[tmp['size'] == size]
                    plt.figure()
                    ax = sns.lineplot(x="th", 
                                      y="round_correct", 
                                      style = 'dim', 
                                      hue = 'dim',
                                      palette=sns.color_palette("Greens_d", len(dim_set)),
                                      data=tmp, 
                                      markers=True
                                      )
                    plt.xticks(np.arange(0.0, 1.05, 0.2))
                    plt.title('th_' + 'graph' + str(graph) + '_algo' + str(algo) + '_size' + str(size))
                    plt.yticks(np.arange(0.0, 1.05, 0.2))
                    plt.legend(fontsize=20)
                    plt.xlabel("Target awareness rate, htrg", fontsize=24)
                    plt.ylabel("Accuracy, R", fontsize=24)
                    plt.tick_params(labelsize=20)
                    plt.tight_layout()
                    plt.savefig('../th_' + 'graph' + str(graph) + '_algo' + str(algo) + '_size' + str(size) + '.png')
                    
       
#algo
if len(algorithm_set) >= 1:
    loop_num = len(size_set) * len(dim_set) * len(graph_set)
    with tqdm(total=loop_num) as pbar:
        for graph in graph_set:
            for dim in dim_set:
                for size in size_set:
                    pbar.update(1)
                    tmp = df_th[df_th['graph'] == graph]
                    tmp = tmp[tmp['dim'] == dim]
                    tmp = tmp[tmp['size'] == size]
                    plt.figure()
                    ax = sns.lineplot(x="th", 
                                      y="round_correct", 
                                      style = 'algo', 
                                      hue = 'algo',
                                      palette=sns.color_palette("Greys_d", len(algorithm_set)),
                                      data=tmp, 
                                      markers=True
                                      )
                    plt.xticks(np.arange(0.0, 1.05, 0.2))
                    plt.title('th_' + 'graph' + str(graph) + '_dim' + str(dim) + '_size' + str(size))
                    plt.yticks(np.arange(0.0, 1.05, 0.2))
                    plt.legend(fontsize=20)
                    plt.xlabel("Target awareness rate, htrg", fontsize=24)
                    plt.ylabel("Accuracy, R", fontsize=24)
                    plt.tick_params(labelsize=20)
                    plt.tight_layout()
                    plt.savefig('../th_' + 'graph' + str(graph) + '_dim' + str(dim) + '_size' + str(size) + '.png')
                    