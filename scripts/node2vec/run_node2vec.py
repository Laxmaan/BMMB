from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed

def process_file(args):
    MAIN_FILE, OUTPUT_ROOT, fpath,dim = args
    fname_stem = fpath.stem

    path = '/'.join(fpath.parent.parts[4:])

    outfpath = OUTPUT_ROOT / path / f'{fname_stem}.emb'

    return f'python {MAIN_FILE} --input {fpath} --output {outfpath} --dimensions {dim} --directed'

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-k", help="value of k",type=int,default=6)
    parser.add_argument("-d",help="embedding dimensions",type=int,default=128)

    args = parser.parse_args()

    DATA_ROOT = Path("datasets") / 'processed' / f'{args.k}' / 'node2vec'

    OUTPUT_ROOT = Path('scripts/node2vec/emb')/f'{args.k}'
    MAIN_FILE = Path('scripts/node2vec/src/main.py')
    files = [x for x in DATA_ROOT.glob("**/*.edgelist")]

    lines = Parallel(n_jobs=-1, verbose=50)(
                                                delayed(process_file)( (MAIN_FILE, OUTPUT_ROOT, files[i], args.d) ) for i in range(len(files)) 
                                            )
    
    with open(f'exec_node2vec_{args.k}.sh','w') as f:
        f.write('\n'.join(lines))
