from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

if __name__=="__main__":
    N2V_ROOT = Path('scripts/node2vec/emb')
    Ks = [4,6,9]

    for k in Ks:
        DATA_ROOT = N2V_ROOT / f'{k}'
        files = DATA_ROOT.glob("**/*.emb")
        out = defaultdict(list)
        dim = None
        for f in files:
            identifier = f.stem
            label = f"{f.parent.parts[4]}/{f.parent.parts[5]}"
            arr = np.loadtxt(f,delimiter=',')
            out[label].append([int(identifier)] + list(arr))
            if not dim:
                dim = len(arr)
        cols = ['idx'] + [f"x_{i}" for i in range(dim)]

        for label,df in out.items():
            df = pd.DataFrame(df, columns=cols)
            df = df.sort_values(["idx"])
            print(f"k {k} label :{label} len {len(df)}")
            output_path = DATA_ROOT / Path(label+".csv")

            Path(output_path).parent.mkdir(exist_ok=True,parents=True)
            df.to_csv(output_path, index=None)
        


