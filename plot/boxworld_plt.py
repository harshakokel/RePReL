import pickle
from datetime import date
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


N = 1
domain = "boxworld"

Black = "#34495e"
Blue = "#3592F0"
GREEN = "#4BAD6C"
RED = "#e74c3c"
Violet = "#B148F7"
yellow = "#DEBE1F"

folder="data/icaps2021_results/"
dfs = []


for r in ["run1", "run2", "run3", "run4", "run5"]:
    for m in ["RePReL+T", "RePReL", "trl+T","trl"]:
        dfs.append(pd.read_csv(f'{folder}/{r}/boxworld-{m}.csv', delimiter=" "))

data = pd.concat(dfs, axis=0 )

def plot_barchart(data, domain='office', taskno=0, legend=False):
    if taskno == 0:
        filtered = (data.task == f'task{taskno}')
        flatui = [RED, GREEN]

        ax = sns.lineplot(x='steps', y='reward', hue='Model', data=data[filtered],
                          palette=sns.color_palette(flatui),
                          legend=False
                          )
    else:
        filtered = (data.task == f'task{taskno}')
        flatui = [RED, RED, GREEN, GREEN]

        ax = sns.lineplot(x='steps', y='reward', hue='Model', data=data[filtered],
                          style='Model',
                          palette=sns.color_palette(flatui),
                          dashes=[(),(2, 2), (),(2, 2)],
                          legend=False
                          )

    if legend:
        if taskno == 0:
            new_labels = ['RePReL','trl']
            plt.legend(labels=new_labels, loc="lower right", fontsize=16)
        else:
            new_labels = ['RePReL+T','RePReL', 'trl+T','trl']
            plt.legend(labels=new_labels, fontsize=14, ncol=2)
    ax.set_axisbelow(True)
    f = lambda x, pos: f'{(x ) / 10 ** 5:,.0f}x$10^5$'
    ax.xaxis.set_major_formatter(FuncFormatter(f))
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.ylim(-50, 5)
    # ax.set_title(F"{domain} - task {taskno + 1}", fontsize=20)
    plt.xlabel('steps in environment', fontsize=16)
    if taskno == 0:
        plt.ylabel("episode reward", fontsize=16)
    else:
        ax.set(ylabel=None)
    plt.savefig(F"{folder}/{domain}_task{taskno}.png", dpi=300)
    plt.show()

plot_barchart(data, domain,0, True)
plot_barchart(data, domain,1, True)
plot_barchart(data, domain,2, True)
