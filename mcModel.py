import random

import networkx as nx
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

#New version of the model that actually follows the math.
# A complete graph G(V, E, w)  weighted, undireceded and connected
# V = [n] (users) E = [m] (edges) w = (weights)

# Friedkin Johnson equilbruim for the expressed opinion of user i
# INPUT: Graph G, and inate opinion s
def FreidkinJohnsonEquilibruim(G, s):
    L = nx.laplacian_matrix(G).astype(float) 
    I = identity(L.shape[0])
    return spsolve(I + L, s)

# Returns the disagreement amongst nodes which is how much their opinion differs of the neighborhood
# INPUT: Graph g, opinion z
def disagreement(G, z):
    return sum((w) * (z[u] - z[v])**2 for u, v, w in G.edges(data = 'weight'))

# This is the opionion of each user (not their inate one that is s)
# INPUT: the opinion z
def opinion(z):
    z_spread = z - np.mean(z)
    return np.dot(z_spread, z_spread)

# Node stress is made up of the disagreement amongst the whole network and that of 
# each nodes inate opinion
# INPUT: Graph g, inate opion s
def nodeStress(G, s):
    z = FreidkinJohnsonEquilibruim(G, s)
    return disagreement(G, z) + opinion(z), z

def opinionUpdate(G, z):
    G_prime = G  # create network
    G_prime.add_nodes_from(G.nodes())
    node_i = random.choice(list(G.prime.nodes()))
    n = len(z)
    op2=0
    w2 = 0
    for j in range(n):
        edgeWeight = G.prime[node_i][i].get(w)
        j_z = z[j]
        op2 = op2+(z[j] * edgeWeight)
        w2 = w2 + edgeWeight
    z_prime = (z[node_i]+op2)/(1+w2)
    G_prime.nodes[node_i]= z_prime

    bernouli_p= 1-abs(z[node_i]-z_prime)

    if random.Random() < bernouli_p
        return G_prime
    else
        return G

if __name__=="__main__":
    n = 20 # number of users
    t = 100 # number of time steps
    z_init = np.random.rand(1)
    G = nx.complete_graph(n)
    
