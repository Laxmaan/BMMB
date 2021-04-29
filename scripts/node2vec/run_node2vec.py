from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed
import src.node2vec as node2vec
import numpy as np
import networkx as nx
from pathlib import Path
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = ArgumentParser(description="Run node2vec.")
    parser.add_argument("-k", help="value of k",type=int,default=6)
    parser.add_argument('--roar', help='usage on Roar systems', action='store_true')



    parser.add_argument("-d",'--dimensions', type=int, default=128,help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--mers',help="number of mers samples",type=int,default=200)
    parser.add_argument('--sars',help="number of sars samples",type=int,default=2300)
    parser.add_argument('--flu',help="number of flu samples",type=int,default=2500)

    return parser.parse_args()

def read_graph(infile):
    '''
    Reads the input network in networkx.
    '''

    G = nx.read_edgelist(infile, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1


    return G

def learn_embeddings(walks,outfile):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    Path(outfile).parent.mkdir(exist_ok=True,parents = True)
    emb = model.wv.mean(axis=0)
    np.savetxt(outfile, emb, delimiter=',')
    return

def process_file(all_args):
    OUTPUT_ROOT, fpath,args = all_args
    fname_stem = fpath.stem

    path = '/'.join(fpath.parent.parts[4:])

    outfpath = OUTPUT_ROOT / path / f'{fname_stem}.emb'

    nx_G = read_graph(fpath)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks,outfpath)


if __name__ == '__main__':
    
 
    args = parse_args()

    DATA_ROOT = Path("datasets") / 'processed' / f'{args.k}' / 'node2vec'

    OUTPUT_ROOT = Path('scripts/node2vec/emb')/f'{args.k}'
    files = [x for x in DATA_ROOT.glob("**/*.edgelist")]

    

    counts = {'mers':args.mers,'sars':args.sars,'influenza':args.flu}
    newlines = []
    for fname in files:
        line = str(fname)
        if 'mers' in line:
            if counts['mers'] > 0:
                newlines.append(line)
                counts['mers'] -= 1
        elif 'sars' in line:
            if counts['sars'] > 0:
                newlines.append(line)
                counts['sars'] -= 1
        elif 'influenza' in line:
            if counts['influenza'] > 0:
                newlines.append(line)
                counts['influenza'] -= 1

    print(f"Tasks :{len(newlines)}")

    Parallel(n_jobs=-1, verbose=50)(
                                                delayed(process_file)( (OUTPUT_ROOT, files[i],args) ) for i in range(len(newlines)) 
                                            )
