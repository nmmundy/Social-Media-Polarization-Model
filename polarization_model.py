import networkx as nx
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

## add in alhpha and beta parameters:
## exponential for the probabilities
## add in alhpha and beta parameters:
## exponential for the probabilities
##Comments on the code:
    ##It is quite basic and with the current set up it almost alwasy adds a new edge 
    ##since the probabily with n = 5 and 15 time steps and the 1 values is polarizing while 0 would not
    ##this currently biases gettig more polarized as the polarizatio value tends to increase at the end 
    ## of all time steps however it has a few increases within as one may expect with a social
    ##network people can be easily or not easily swayed or if they have friends whos innate opinions
    ## are opposite to their own but they still follow them that can infuence the polarization
    ## from a mathematical standpoint.
    ##Newtork polarization = polarization + disagreements


#Uses Friedkin-Johnson opinion model from paper in overleaf notes
#Uses Friedkin-Johnson opinion model from paper in overleaf notes
#Each node maintaines an inate opinion and the update to a node is based on 
#expressed opinion z_i.
def opinion_dynamics(G, s):
    L = nx.laplacian_matrix(G).astype(float) #Laplacian from paper
    L = nx.laplacian_matrix(G).astype(float) #Laplacian from paper
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
    #return polarization(z) + disagreement_weights(G, z), z
    return -polarization(z), z

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
            p = max(0, 1 - abs(z_old[i] / z_old[j] - 1))
            #print(f"p = {p}")
            p = np.clip(p, 0, 1)
            if np.random.rand() < p:
                G_new.add_edge(i, j, weight=1.0)
        
        #removes an edge aka removes a following
        #Seems to somewhat work, the plot of the graph just looks kind of odd and random, but 
        # i think that is just due to how matplotlib is placing the nodes, may need to add a grid so its
        # more stuctured
        for i in range(n):
            for j in range(i + 1, n):
                if not G_new.has_edge(i, j):  #skips non‑existent edges
                    continue
                # 1-p from above? might be more right as not 100% sure they add to 1 which is bad
                p = abs(z_old[j] - z_old[i])        # fixed index & formula
                if np.random.rand() < np.clip(p, 0, 1):
                    G_new.remove_edge(i, j)

        
        #removes an edge aka removes a following
        #Seems to somewhat work, the plot of the graph just looks kind of odd and random, but 
        # i think that is just due to how matplotlib is placing the nodes, may need to add a grid so its
        # more stuctured
        for i in range(n):
            for j in range(i + 1, n):
                if not G_new.has_edge(i, j):  #skips non‑existent edges
                    continue
                # 1-p from above? might be more right as not 100% sure they add to 1 which is bad
                p = abs(z_old[i] - z_old[j])       # fixed index & formula
                if np.random.rand() < np.clip(p, 0, 1):
                    G_new.remove_edge(i, j)

    return G_new

def metropolis_step(G, z, s, alpha=1.0, proposals_per_step=10):
    G_new = G.copy()
    n = len(z)
    for _ in range(proposals_per_step):
        i, j = np.random.choice(n, 2, replace=False)
        G_prime = G_new.copy()

        if G_new.has_edge(i, j):
            G_prime.remove_edge(i, j)
        else:
            G_prime.add_edge(i, j, weight=1.0)

        E_old, _ = polarization_adjustments(G_new, s)
        E_new, _ = polarization_adjustments(G_prime, s)
        delta_E = E_new - E_old
        accept_prob = min(1, np.exp(alpha * delta_E))

        if np.random.rand() < accept_prob:
            G_new = G_prime

    return G_new

def metropolis_step(G, z, s, alpha = 5.0):
    G_prime = G.copy()
    n = len(z)
    i, j = np.random.choice(n, 2, replace=False)

    if G.has_edge(i, j):
        G_prime.remove_edge(i, j)
    else:
        G_prime.add_edge(i, j, weight=1.0)

    E_old, _ = polarization_adjustments(G, s)
    E_new, _ = polarization_adjustments(G_prime, s)

    delta_E = E_new - E_old
    accept_prob = min(1, np.exp(-alpha * delta_E))

    if np.random.rand() < accept_prob:
        return G_prime
    else:
        return G

def time_evolution(z0, t, alpha):
def time_evolution(z0, t, alpha, proposals, metropolis):
    n = len(z0)

    #Initialize the network
    G = nx.complete_graph(n)
    nx.set_edge_attributes(G, 1., 'weight')
    nx.set_edge_attributes(G, 1., 'weight')

    fig_energy, ax_energy = plt.subplots(figsize=(6, 3))
    polarization_trace = []

    #Plot stuff
    #Plot stuff
    plt.ion() #plot interactivity
    fig = plt.figure(figsize=(5, 4))

    for time in range (t):
        I, z = polarization_adjustments(G, z0)
        print(f"t = {time}, Network Polarization = {I:.4f}")
        draw_graph(G, z, time)
        G = metropolis_step(G, z, z0, alpha)
        #print(f"t = {time}, Network Polarization = {I:.4f}")
        polarization_trace.append(I)
        draw_graph(G, z, time)
        draw_energy(polarization_trace, ax_energy)
        if metropolis:
            G = metropolis_step(G, z, z0, alpha, proposals)
        else:
            G = markov_process(G, z)

        ##more plot stuff
        plt.ioff()

        ##more plot stuff
        plt.ioff()
        #plt.show(block=True)
        #plt.show()

def draw_graph(G, z, step):
def draw_graph(G, z, step):
    plt.clf()
    pos = nx.spring_layout(G, seed=42)   # same layout every frame
    # node colors are the expressed opinions (blue color = low, yellow color = higher)
    # node colors are the expressed opinions (blue color = low, yellow color = higher)
    nx.draw_networkx_nodes(G, pos,
        node_color=z, cmap='plasma', vmin=0, vmax=1,
        node_size=300, linewidths=1, edgecolors='black')
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.0)
    plt.title(f"Step {step}")
    plt.axis('off')
    plt.pause(0.3) 
    plt.pause(0.1) 

def draw_energy(energy_trace, ax):
    ax.clear()
    ax.plot(energy_trace, label='Polarization')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Polarization")
    ax.set_title("Polarization Over Time")
    ax.legend()
    ax.grid(True)
    plt.pause(0.01)

if __name__ == "__main__":
    np.random.seed(32)
    n = 20 # number of users
    t = 100 # time steps
    n = 25 # number of users
    t = 100 # time steps
    z0 = np.random.rand(n)
    alpha = 1.
    proposals = 10
    #picking which function metropolis or our original mc function
    metropolis = True

    #Gr_i, z_i, I_i = 
    time_evolution(z0, t, alpha=5.0)
    time_evolution(z0, t, alpha, proposals, metropolis)