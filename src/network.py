"""
Class to generate a network with velocity/distance based delays
    A: Adjacency matrix for the newtork
    D: Delay matrix - time for signal transmission between nodes.
    types: line, lattice, ring, full, smallworld

"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self, config):
        # Set network parameters
        self.config = config
        self.N = config["N"]
        
        # Generate network
        self.network = self.make_network()
        self.A = self.get_adjacency_matrix()
        self.D = self.get_delay_matrix()
        
    def get_adjacency_matrix(self):
        """ Creates NxN connectivity matrix from networkx graph"""
        A = nx.to_numpy_array(self.network)
        return A
    
    def get_delay_matrix(self):
        """ Create an NxN matrix of delays between nodes
            Assumes constant signal velocity
            Assumes constant node distance
            Derives distances from network structure"""
        d = self.config["node_distance"]
        v = self.config["signal_velocity"]
        return self.A * (d / v)
    
    def make_network(self):
        topology = self.config["topology"]
        if topology == "line":
            return self.make_line_network()
        elif topology == "lattice":
            return self.make_lattice_network()
        elif topology == "ring":
            return self.make_ring_network()
        elif topology == "full":
            return self.make_full_network()
        elif topology == "smallworld":
            return self.make_smallworld_network()
        else:
            raise ValueError(f"Network topology {topology} not recognised")

    def make_line_network(self):
        """ Network of N nodes connected in a line"""
        return nx.path_graph(self.N)
    
    def make_lattice_network(self):
        """ Network of N nodes connected in a square lattice """
        # Assuming a roughly square grid that fits N nodes
        side = int(np.ceil(np.sqrt(self.N)))
        G = nx.grid_2d_graph(side, side)
        # Relabel nodes to integers 0 to N-1 and trim excess nodes
        G = nx.convert_node_labels_to_integers(G)
        G.remove_nodes_from(list(G.nodes())[self.N:])
        return G

    def make_ring_network(self):
        """Generate circulant network with p% neighbouring connections"""
        p = self.config.get("p", 0.1) # Default to 10% if not provided
        k = int(p * self.N)
        offsets = [i+1 for i in range(k)]
        return nx.circulant_graph(self.N, offsets) 
    
    def make_full_network(self):
        """ Network of N nodes fully connected"""
        return nx.complete_graph(self.N)

    def make_smallworld_network(self):
        """Generate Watts-Strogatz network with p% neighbouring connections"""
        p = self.config.get("p", 0.1)
        rewire_prob = self.config.get("rewire_prob", 0.05)
        k = int(p * self.N)
        k = max(2, k) 
        return nx.watts_strogatz_graph(self.N, k, rewire_prob)
    
    def plot_network_matrix(self, mode="Adjacency"):
        if mode == "Adjacency":
            matrix = self.A
        elif mode == "Delay":
            matrix = self.D
        else:
            raise ValueError(f"Mode {mode} not recognised")
        plt.imshow(matrix)
        plt.title(f"{mode} Matrix")
        plt.xlabel("Node Index")
        plt.ylabel("Node Index")
        plt.tight_layout()
        plt.show()