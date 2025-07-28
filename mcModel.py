import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

class MarkovChainPolarizationModel:
    def __init__(self, n, beta=2.0, sigma=0.1, seed=None):
        self.n = n # number of nodes
        self.beta = beta # sensitivity to opinion updates parameter
        self.sigma = sigma # standard deviation for edge weights
        self.rng = np.random.default_rng(seed) # random number generator

        self.z = self.rng.uniform(0, 1, size=n) # opinions of each node at time step t

        w = self.rng.uniform(0, 1, size=(n, n)) # weights of edges at current time
        np.fill_diagonal(w, 0)
        self.w = w / w.sum(axis=1, keepdims=True)

    def opinionUpdate(self):
        z_prime = np.zeros_like(self.z)

        for t in range(self.n):
            ## Below two lines represent Equation 4 in the overleaf
            weighted_opinions = np.sum(self.w[t] * self.z) 
            z_i = (self.z[t] + weighted_opinions) / (1 + np.sum(self.w[t]))
            
            ## Next line is Equation 5, the probabiliy
            p_i = np.exp(-self.beta * abs(z_i - self.z[t]))

            if (self.rng.random() < p_i):
                z_prime[t] = z_i

        self.z = z_prime

    def edgeWeightUpdate(self):
        w_prime = np.zeros_like(self.w)
        for i in range(self.n):
            for j in range(self.n):
                if i !=j: 
                    ## mu in equation 8 in overleaf document
                    mu = 1 - abs(self.z[i] - self.z[j]) 
                    # equation 8 from overleaf
                    w_ij = self.rng.normal(mu, self.sigma)
                    w_prime[i,j] = max(0, w_ij)

            if w_prime[i].sum() > 0:
                w_prime[i] /= w_prime[i].sum() # normalization of weights
        
        self.w = w_prime

    def timeStep(self):
        self.opinionUpdate()
        self.edgeWeightUpdate()

    def runModel(self, t = 20):
        opinions = [self.z.copy()]
        weights = [self.w.copy()]
        
        for _ in range(t):
            self.timeStep()
            opinions.append(self.z.copy())
            weights.append(self.w.copy())
        
        return np.array(opinions), weights
    
def visualization(opinions, weights, interval=500):
    n = opinions.shape[1]
    t = opinions.shape[0]
    Graph = nx.complete_graph(n)
    position = nx.circular_layout(Graph)

    fig, (network, opinion_traj) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))

    lines = []
    for i in range(n):
        (line,) = opinion_traj.plot([], [], color=colors[i], label=f"Node {i}")
        lines.append(line)
    opinion_traj.set_xlim(0, t-1)
    opinion_traj.set_ylim(0, 1)
    opinion_traj.set_xlabel("Time Step")
    opinion_traj.set_ylabel("Opinion Evolution")
    opinion_traj.legend(loc="upper right")
    opinion_traj.grid(True, linestyle='--', alpha=0.5)

    def realTimeUpdate(frame):
        network.clear()
        curr_opinions = opinions[frame]
        curr_weights = weights[frame]

        node_colors = plt.cm.coolwarm(curr_opinions)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_weights = [curr_weights[i, j] for i, j in edges]

        nx.draw_networkx_nodes(Graph, position, node_color=node_colors, node_size=400, ax=network)
        nx.draw_networkx_edges(Graph, position, edgelist=edges,
                            width=[3 * w for w in edge_weights],
                            edge_color='gray', alpha=0.7, ax=network)
        nx.draw_networkx_labels(Graph, position, {i: str(i) for i in range(n)}, ax=network)
        network.set_title(f"Network at Time {frame}")
        network.axis('off')

        # Update opinion trajectories
        for i in range(n):
            lines[i].set_data(range(frame + 1), opinions[:frame + 1, i])

    ani = FuncAnimation(fig, realTimeUpdate, frames=t, interval=interval, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani

if __name__ == "__main__":
    model = MarkovChainPolarizationModel(n=20, beta=-.5, sigma=0.05, seed=42)
    opinions, weights = model.runModel(t=30)
    visualization(opinions, weights)