import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

class MarkovChainPolarizationModel:
    def __init__(self, n, eta=2.0, beta=2.0, alpha=2.0, sigma=0.1, seed=None):
        self.n = n # number of nodes
        self.eta = eta # sensitivity to opinion updates parameter
        self.sigma = sigma # standard deviation for edge weights
        self.rng = np.random.default_rng(seed) # random number generator
        self.beta = beta # parameter for the beta distribution
        self.alpha = alpha # parameter for beta distribution
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
            p_i = np.exp(-self.eta * abs(z_i - self.z[t]))

            if (self.rng.random() < p_i):
                z_prime[t] = z_i

        self.z = z_prime
    '''
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
        '''
    ## Revised edge weights now company has influence
    ## self, platform input, target opinion zPrime, strengh of company bias (0,1)
    def edgeWeightUpdate(self, platformInfluence=True, zPrime=1, gamma=.5, zeta=10):
        concentration = 5
        w_new = np.zeros_like(self.w)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    userOpinion = 1 - abs(self.z[i] - self.z[j])

                    if platformInfluence:
                        companyBias = 1 + gamma * np.exp(-zeta * abs(self.z[j] - zPrime))
                        mu = companyBias + userOpinion
                    else:
                        mu = userOpinion

                    mu = np.clip(mu, 0.001, .999)
                    alpha = mu * concentration
                    beta_ = (1 - mu) * concentration

                    w_ij = self.rng.beta(alpha, beta_)
                    w_new[i,j] = w_ij

                if w_new[i].sum() > 0:
                    w_new[i, j] = w_ij
            
            self.w = w_new
    
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
    
# --- REVISED VISUALIZATION FUNCTION ---
def visualization(opinions, weights, interval=500):
    n = opinions.shape[1]
    t = opinions.shape[0]

    # Create the graph object, which will be updated
    Graph = nx.Graph()
    Graph.add_nodes_from(range(n))

    # We use a stable, initial layout for the y-coordinates
    initial_y_pos = {i: 2 * (i / (n - 1)) - 1 for i in range(n)}

    fig, (network, opinion_traj) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))

    # --- Opinion Trajectory Setup (unchanged) ---
    lines = []
    for i in range(n):
        (line,) = opinion_traj.plot([], [], color=colors[i], label=f"Node {i}")
        lines.append(line)
    opinion_traj.set_xlim(0, t - 1)
    opinion_traj.set_ylim(0, 1)
    opinion_traj.set_xlabel("Time Step")
    opinion_traj.set_ylabel("Opinion Evolution")
    opinion_traj.legend(loc="upper right")
    opinion_traj.grid(True, linestyle='--', alpha=0.5)

    # --- Network Visualization Setup ---
    network.set_title("Network Visualization")
    network.set_xticks([])
    network.set_yticks([])
    network.set_xlim(-0.1, 1.1)
    network.set_ylim(-1.1, 1.1)

    # --- The Animation Update Function (KEY CHANGES HERE) ---
    def update(frame):
        # Clear the network plot for the new frame
        network.clear()

        curr_opinions = opinions[frame]
        curr_weights_matrix = weights[frame]

        # Update the graph's edges with the current weights
        Graph.clear_edges()
        for i in range(n):
            for j in range(i + 1, n):
                weight = curr_weights_matrix[i, j]
                if weight > 0:  # Only add edges with a non-zero weight
                    Graph.add_edge(i, j, weight=weight)

        # KEY CHANGE: Calculate the layout using nx.spring_layout
        # The 'weight' attribute of the edges will be used to determine spring strength.
        # k: ideal distance between nodes.
        # iterations: one step of the layout algorithm per frame for smooth movement.
        pos = nx.spring_layout(Graph, pos=None, weight='weight', k=0.5, iterations=5, scale=1)

        # Set the x-position to opinion value and the y-position to a constant for a clearer visualization
        for node in Graph.nodes():
            pos[node][0] = curr_opinions[node]
            # You can uncomment the line below to give the layout a starting y-position
            # pos[node][1] = initial_y_pos[node]

        node_colors = plt.cm.coolwarm(curr_opinions)

        # Draw the network with the new layout
        edge_widths = [d['weight'] * 5 for u, v, d in Graph.edges(data=True)]

        nx.draw_networkx_nodes(Graph, pos, node_color=node_colors, node_size=400, ax=network)
        nx.draw_networkx_edges(Graph, pos, width=edge_widths, edge_color='gray', alpha=0.7, ax=network)
        nx.draw_networkx_labels(Graph, pos, {i: str(i) for i in range(n)}, ax=network)

        network.set_title(f"Time Step: {frame}")
        network.axis('off')

        # Set stable limits to prevent the plot from resizing
        network.set_xlim(-0.1, 1.1)
        network.set_ylim(-1.1, 1.1)

        # Update opinion trajectories
        for i in range(n):
            lines[i].set_data(range(frame + 1), opinions[:frame + 1, i])

        return lines + list(network.get_children())

    # --- Create the Animation ---
    ani = FuncAnimation(fig, update, frames=t, interval=interval, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

if __name__ == "__main__":
    model = MarkovChainPolarizationModel(n=15, beta=8, sigma=10, seed=42)
    opinions, weights = model.runModel(t=30)
    visualization(opinions, weights)