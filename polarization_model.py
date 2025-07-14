import networkx as nx
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

##Comments on the code:
    ##It is quite basic and with the current set up it almost alwasy adds a new edge 
    ##since the probabily with n = 5 and 15 time steps and the 1 values is polarizing while 0 would not
    ##this currently biases gettig more polarized as the polarizatio value tends to increase at the end 
    ## of all time steps however it has a few increases within as one may expect with a social
    ##network people can be easily or not easily swayed or if they have friends whos innate opinions
    ## are opposite to their own but they still follow them that can infuence the polarization
    ## from a mathematical standpoint.
    ##Newtork polarization = polarization + disagreements


#Uses Friedkin-Johbson opinion model from paper in overleaf notes
#Each node maintaines an inate opinion and the update to a node is based on 
#expressed opinion z_i.
def opinion_dynamics(G, s):
    L = nx.laplacian_matrix(G).astype(float)
    I = identity(L.shape[0])
    return spsolve(I + L, s)

##the below two functions make up the total polarization known as Network Polarization
#this determines the weights on each edge with influences
# aka the disagreements
#computes this equation: Sum(over all edges uv) w_uv(z_u - z_v)^2
def disagreement_weights(G, z):
    return sum((w) * (z[u] - z[v])**2 for u, v, w in G.edges(data = 'weight'))

##computes this equation: sum(i) (z_i - z_bar)^2 -> ||z_i - z_bar||^2
#returns the polarizaton of the overall graph -> spread of opinions compared to average
def polarization(z):
    z_spread = z - np.mean(z)
    return np.dot(z_spread, z_spread)

def polarization_adjustments(G, s):
    z = opinion_dynamics(G, s)
    return polarization(z) + disagreement_weights(G, z), z

def markov_process(G_old, z_old):
    G_new = nx.Graph()
    G_new.add_nodes_from(G_old.nodes())

    n = len(z_old)
    #print("z_old")
    #print(z_old)
    for i in range(n):
        for j in range(i + 1, n):
            if z_old[j] == 0:
                continue
            #probablities 
            #Adding in a new edge with probability specificed in overleaf
            p = 1 - abs(z_old[i] / z_old[j] - 1)
            #print(f"p = {p}")
            p = np.clip(p, 0, 1)
            if np.random.rand() < p:
                G_new.add_edge(i, j, weight=1.0)
    for i in range(n):
        for j in range(i + 1, n):
            if z_old[j] == 1:
                continue
            p = abs(z_old[n] - z_old[i])
            p = np.clip(p, 0, 1)
            if np.random.rand() < p:
                G_new.remove_edge(i, j)
    return G_new

def time_evolution(z0, t):
    n = len(z0)
    Gr_i = []
    z_i = []
    I_i = []

    #Initialize the network
    G = nx.complete_graph(n)
    nx.set_edge_attributes(G, 1, 'weight')

    plt.ion() #plot interactivity
    fig = plt.figure(figsize=(5, 4))

    for time in range (t):
        I, z = polarization_adjustments(G, z0)
        Gr_i.append(G.copy())
        z_i.append(z.copy())
        I_i.append(I)
        print(f"t = {time}, Network Polarization = {I:.4f}")
        #draw_graph(G, z, time)
        G = markov_process(G, z)

        #plt.ioff()
        #plt.show()
    #return Gr_i, z_i, I_i

#def draw_graph(G, z, step):
    plt.clf()
    pos = nx.spring_layout(G, seed=42)   # same layout every frame
    # node colors = expressed opinions (bluer low, yellower high)
    nx.draw_networkx_nodes(G, pos,
        node_color=z, cmap='plasma', vmin=0, vmax=1,
        node_size=300, linewidths=1, edgecolors='black')
    # thin gray edges
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.0)
    plt.title(f"Step {step}")
    plt.axis('off')
    plt.pause(0.3)   # short delay so the window updates

if __name__ == "__main__":
    np.random.seed(32)
    n = 5
    t = 15
    z0 = np.random.rand(n)

    #Gr_i, z_i, I_i = 
    time_evolution(z0, t)