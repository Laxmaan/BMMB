import pandas as pd
import tensorflow as tf
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    DATA_ROOT = Path('datasets/embs')
    algos = ['node2vec','graph2vec']
    K = [4,6,9]

    labelmap = {4:{'covid':0,'influenza':1},
                6:{'sars':0,'mers':1,'influenza':2},
                9:{'sars':0,'mers':1,'influenza':2}}
    for algo in algos:
        for k in K:
            ROOT = DATA_ROOT / algo / str(k)
            print(algo,k)
            csvs = [x for x in ROOT.glob("*.csv")]
            vals = []
            labels = []
            for csv in csvs:
                df = pd.read_csv(csv)
                df = df.drop(df.columns[0], axis=1)
                vals.append(df.values)

                labels.extend([  labelmap[k][csv.stem]  ]*len(df))
            labels = np.array(labels)
            
            vals = np.vstack(vals)
            
