from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed

def process_file(args):
    MAIN_FILE, OUTPUT_ROOT, fpath = args
    fname_stem = fpath.stem

    path = '/'.join(fpath.parent.parts[4:])

    outfpath = OUTPUT_ROOT / path / f'{fname_stem}.emb'

    return f'python {MAIN_FILE} --input {fpath} --output {outfpath}'

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-k", help="value of k",type=int,default=6)

    args = parser.parse_args()

    DATA_ROOT = Path("datasets") / 'processed' / f'{args.k}' / 'node2vec'

    OUTPUT_ROOT = Path('scripts/node2vec/emb')
    MAIN_FILE = Path('scripts/node2vec/src/main.py')
    files = [x for x in DATA_ROOT.glob("**/*.edgelist")]

    lines = Parallel(n_jobs=-1, verbose=50)(
                                                delayed(process_file)( (MAIN_FILE, OUTPUT_ROOT, files[i]) ) for i in range(len(files)) 
                                            )
    
    with open('exec_node2vec.sh','w') as f:
        f.write('\n'.join(lines))
