"""Graph2Vec module."""

import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathlib import Path
from collections import defaultdict

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.stem
    graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())

    if "features" in data.keys():
        features = data["features"].items()
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v in features}
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(OUTPUT_ROOT, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    
    out = defaultdict(list)
    for f in files:
        identifier = f.stem
        label = f"{f.parent.parts[4]}/{f.parent.parts[5]}"
        out[label].append([int(identifier)] + list(model.docvecs["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]

    for label,df in out.items():
        df = pd.DataFrame(df, columns=column_names)
        df = df.sort_values(["type"])

        output_path = OUTPUT_ROOT / Path(label+".csv")

        Path(output_path).parent.mkdir(exist_ok=True,parents=True)
        df.to_csv(output_path, index=None)

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """

    DATA_ROOT = Path("datasets") / 'processed' / f'{args.k}' / 'node2vec'

    OUTPUT_ROOT = Path('scripts/graph2vec/emb')/f'{args.k}'
    graphs = [x for x in DATA_ROOT.glob("**/*.edgelist")]
    counts = {'mers':args.mers,'sars':args.sars,'influenza':args.flu}
    newlines = []

    for fname in graphs:
        line = str(fname)
        if 'mers' in line:
            if counts['mers'] > 0:
                newlines.append(fname)
                counts['mers'] -= 1
        elif 'sars' in line:
            if counts['sars'] > 0:
                newlines.append(fname)
                counts['sars'] -= 1
        elif 'influenza' in line:
            if counts['influenza'] > 0:
                newlines.append(fname)
                counts['influenza'] -= 1

    print(f"\nFiltered tasks = {len(newlines)}")
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(newlines))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)

    save_embedding(OUTPUT_ROOT, model, newlines, args.dimensions)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
