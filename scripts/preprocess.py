from Bio import SeqIO
from pathlib import Path
import json
import re
from joblib import Parallel, delayed

DATA_ROOT = Path("datasets")
VOCAB = Path('vocab')

RAW = DATA_ROOT / 'raw'

PROCESSED = DATA_ROOT  /'processed'

labels = [x.name for x in RAW.glob("*")]
files = [x.name for x in RAW.glob("**/**/*")]

def seq_to_db(seq,k,vocab):
    seq = str(seq).upper()
    if not bool(re.match('^[ATGC]+$', seq)):
        return None,None
    n = len(seq)
    edgelist = set()
    edges = []
    for i in range(n-k+1):
        kmer = seq[i:i+k]
        u = kmer[:-1]
        v = kmer[1:]
        if u not in vocab:
            vocab[u] = len(vocab)
        if v not in vocab:
            vocab[v] = len(vocab)

        edgelist.add(f'{vocab[u]} {vocab[v]}')
        edges.append([vocab[u],vocab[v]])

    edges ={'edges':edges}
    return edgelist,edges



def process_record(args):
    idx,OUT_DIR,JSON_DIR,record,k,vocab = args
    g,g1 = seq_to_db(record.seq, k, vocab)
    if not g:
        return

    with open(OUT_DIR / f'{idx}.edgelist','w') as edgefile:
        edgefile.write('\n'.join(g))

    with open(JSON_DIR / f'{idx}.json','w') as edgefile:
        json.dump(g1,edgefile)
    
    
def make_db_graph(filename, OUT_DIR,JSON_DIR, k,vocab):
    ctr = 0
    with open(filename) as f:
        Gs = Parallel(n_jobs=-1, verbose=50)(delayed(process_record)((i,OUT_DIR,JSON_DIR,record,k,vocab)) for i,record in enumerate(SeqIO.parse(f,"fasta")))

    

def make_vocab(k):
    vocab = {}
    alpha = ['A','T','G','C']

    def helper(curr,vocab):
        if len(curr) == k - 1:
            vocab[curr] = len(vocab)
        else:
            for a in alpha:
                helper(curr+a,vocab)

    helper('',vocab)
    vocab_file = VOCAB / f'k_{k}.json'
    with open(vocab_file,'w') as f:
            json.dump(vocab, f)





    
def process_label(label,k=6):
    path = RAW / label
    files = list(path.glob("*"))

    vocab_file = VOCAB / f'k_{k}.json'
    if not vocab_file.exists():
        make_vocab(k)
    with open(vocab_file) as f:
        vocab = json.load(f)

    for fname in files:
        data_dir = fname.stem
        print(f'\nprocessing {fname}\n')
        OUT_DIR = PROCESSED / f'{k}' / 'node2vec' / label / data_dir
        OUT_DIR.mkdir(exist_ok = True, parents = True)
        JSON_DIR = PROCESSED / f'{k}' / 'graph2vec' / label / data_dir
        JSON_DIR.mkdir(exist_ok = True, parents = True)
        
        make_db_graph(fname, OUT_DIR,JSON_DIR, k, vocab)
        
        


#read_fasta(RAW/ 'Covid'/ 'mers.fna')
for label in labels:
    process_label(label)