"""
Functions to generate and analyse networks
"""
import networkx as nx


def get_circulant_network(N, p):
    """Generate circulant network with p% neighbouring connections, symmetrically distributed"""
    k = int(p * N)
    offsets = [i+1 for i in range(k)]
    A = nx.circulant_graph(N, offsets) 
    return nx.to_numpy_array(A)


def get_smallworld_network(N, p, rewire_prob):
    k = int(p * N)
    A = nx.watts_strogatz_graph(N, k, rewire_prob)
    return nx.to_numpy_array(A)