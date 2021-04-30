from pathlib import Path
import pandas as pd






if __name__ == '__main__':
    DATA_ROOT = Path('datasets')/'embs'
    SCRIPTS = Path('scripts')
    algos = ['node2vec','graph2vec']
    K = [4,6,9]

    for alg in algos:
        ALG_OUT_DIR = DATA_ROOT / alg
        EMB_DATA_DIR = SCRIPTS / alg / 'emb'
        for k in K:
            DATA_DIR = EMB_DATA_DIR / str(k)
            csvs = [str(x)  for x in DATA_DIR.glob("**/*.csv")]
    
            OUT_DIR = ALG_OUT_DIR / str(k)
            OUT_DIR.mkdir(exist_ok = True,parents = True)
            if k == 4:
                flu = None
                covid = []
                for fname in csvs:
                    df = pd.read_csv(fname)
                    if 'Covid' in fname:
                        covid.append(df)
                    else:
                        flu = df
                covid = pd.concat(covid)
                covid.to_csv(OUT_DIR / 'covid.csv', index=None)
                flu.to_csv(OUT_DIR / 'influenza.csv', index=None)

            else:
                for fname in csvs:
                    df = pd.read_csv(fname)
                    name = Path(fname).name
                    df.to_csv(OUT_DIR / name, index = None)
        
