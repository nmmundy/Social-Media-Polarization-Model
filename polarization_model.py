import networkx as nx
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

def opinion_dynamics(G, s):
    L = nx.laplacian_matrix(G).astype(float)
    I = sp.identity(L.shape[0])
    return spsolve(I + L, s)

#this determines the weights on each edge with influences
#the disagreements
def weights(G, z):
    return sum(w * (z[u] - z[v])**2 for u, v, w in G.edges(data = 'weight'))

def polarization(z):
    z_new = z - np.mean(z)
    return np.dot(z_new, z_new)

def polatization_adjustments(G, s):
    z = opinion_dynamics(G, s)
    return polarization(z) + weights(G, z), z

def markov_process(G_old, z_old, threshold=0.1):
    G_new = nx.Graph()
    G_new.add_nodes_from(G_old.nodes())

    n = len(z_old)
    for i in range(n):
        for j in range(i + 1, n):
            if z_old[j] == 0:
                continue
            #probablities
            p = 1 - abs(z_old[i] / z_old[i] - 1)
            p = np.clip(p, 0, 1)
            if np.random.rand() < p:
                G_new.add_edge(i, j, weights = 1.)
    return G_new

def time_evolution(z0, t):
    n = len(z0)
    Gr_i = []
    z_i = []
    I_i = []

    #Initialize the network
    G = nx.complete_graph(n)
    nx.set_edge_attributes_attributes(G, 1, 'weight')

    for time in range (t):
        I, z = polatization_adjustments(G, z0)
        Gr_i.append(G.copy())
        z_i.append(z.copy())
        I_i.append(I)
        G = markov_process(G, z)

        print(f"t = {t}, PD Index = {I:.4f}")

    return Gr_i, z_i, I_i

np.random.seed(32)
n = 15
t = 20
z0 = np.random.rand(n)

Gr_i, z_i, I_i = markov_process(z0, t)