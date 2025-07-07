import networkx as nx
import numpy as np

def load_karate():
    G = nx.karate_club_graph()
    ground_truth = np.array([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()])
    return G, ground_truth

def load_dolphins():
    G = nx.read_gml('dolphins.gml', label='id')
    ground_truth = None
    return G, ground_truth

def load_football():
    G = nx.read_gml("football.gml")
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    ground_truth = [G.nodes[n].get("value", 0) for n in G.nodes()]
    return G, ground_truth