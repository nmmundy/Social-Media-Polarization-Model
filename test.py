import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class SocialNetworkMarkovChain:
    def __init__(self, N, beta=5.0, sigma=0.1, seed=None):
        """
        N     : Number of nodes
        beta  : Sensitivity to opinion shifts
        sigma : Std dev for weight updates
        """
        self.N = N
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        
        # Initialize opinions and weights
        self.z = self.rng.uniform(0, 1, size=N)  # opinions in [0,1]
        self.w = self._init_weights()
    
    def _init_weights(self):
        """Initialize a weight matrix with rows summing to 1."""
        w = self.rng.uniform(0, 1, size=(self.N, self.N))
        np.fill_diagonal(w, 0)  # No self-loops
        w = w / w.sum(axis=1, keepdims=True)  # Normalize each row
        return w

    def update_opinions(self):
        """Perform one opinion update step with logistic success probability."""
        new_z = self.z.copy()
        for i in range(self.N):
            # Compute proposed opinion
            neighbors = [j for j in range(self.N) if j != i]
            # Ensure neighbors exist to avoid division by zero
            if not neighbors or np.sum(self.w[i, neighbors]) == 0:
                continue
            
            proposed = (self.z[i] + np.sum(self.w[i, neighbors] * self.z[neighbors])) / \
                       (1 + np.sum(self.w[i, neighbors]))
            
            # Logistic-like success probability
            p_success = np.exp(-self.beta * abs(proposed - self.z[i]))
            
            # Bernoulli trial
            if self.rng.random() < p_success:
                new_z[i] = proposed
        
        self.z = new_z

    def update_weights(self):
        """Update edge weights based on opinion similarity (homophily rule)."""
        new_w = np.zeros_like(self.w)
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                mu = 1 - abs(self.z[i] - self.z[j])  # similarity-based mean
                new_w[i, j] = self.rng.normal(loc=mu, scale=self.sigma)
                new_w[i, j] = np.clip(new_w[i, j], 0, 1)  # ensure within [0,1]
            
            row_sum = new_w[i].sum()
            if row_sum > 0:
                new_w[i] /= row_sum
            elif self.N > 1:
                # Fallback if all weights are zero: uniform distribution
                new_w[i] = 1.0 / (self.N - 1)
                np.fill_diagonal(new_w, 0)

        self.w = new_w

    def polarization(self):
        """Compute polarization PG(t)."""
        mean_opinion = np.mean(self.z)
        return np.sum((self.z - mean_opinion)**2)
    
    def disagreement(self):
        """Compute disagreement DG(t)."""
        dg = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N): # Iterate over upper triangle for undirected graph
                # Consider a symmetric interaction for disagreement
                weight = (self.w[i, j] + self.w[j, i]) / 2.0
                dg += weight * (self.z[i] - self.z[j])**2
        return dg

    def step(self, weight_updates=1):
        """Perform one full step of the Markov chain."""
        self.update_opinions()
        for _ in range(weight_updates):
            self.update_weights()
    
    def run_with_trajectories(self, T=100, weight_updates=1):
        """
        Run the chain for T steps, storing opinion trajectories and polarization/disagreement.
        """
        Z = [self.z.copy()]  # store opinion vectors over time
        PG, DG = [self.polarization()], [self.disagreement()]

        for _ in range(T):
            self.step(weight_updates)
            Z.append(self.z.copy())
            PG.append(self.polarization())
            DG.append(self.disagreement())

        return np.array(Z), np.array(PG), np.array(DG)

    def plot_opinion_trajectories(self, Z):
        """
        Plot opinion trajectories over time.
        Z is an array of shape (T+1, N).
        """
        T = Z.shape[0]
        plt.figure(figsize=(8, 5))
        for i in range(Z.shape[1]):
            plt.plot(range(T), Z[:, i], alpha=0.8)
        plt.xlabel("Time step")
        plt.ylabel("Opinion")
        plt.ylim(-0.05, 1.05)
        plt.title("Opinion Trajectories Over Time")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_polarization_disagreement(self, PG, DG):
        """
        Plot polarization and disagreement over time.
        """
        T = len(PG)
        plt.figure(figsize=(8, 5))
        plt.plot(range(T), PG, label='Polarization (PG)', linewidth=2)
        plt.plot(range(T), DG, label='Disagreement (DG)', linewidth=2)
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title("Polarization and Disagreement Over Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    def plot_network(self, title="Network State"):
        """
        Visualize the network with node opinions and edge weights.
        """
        G = nx.from_numpy_array((self.w + self.w.T) / 2.0) # Create a symmetric graph for visualization
        
        pos = nx.spring_layout(G, seed=42)
        node_colors = [self.z[i] for i in G.nodes()]
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Draw the network on the specified axes
        nx.draw(
            G, pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            cmap=plt.cm.coolwarm,
            node_size=500,
            width=edge_widths,
            edge_color='gray',
            vmin=0,
            vmax=1
        )
        
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        # Add the colorbar, specifying the axes to use
        fig.colorbar(sm, ax=ax, label='Opinion', shrink=0.8)
        
        ax.set_title(title)
        plt.show()


# --- Main execution ---
model = SocialNetworkMarkovChain(N=8, beta=100.0, sigma=0.1, seed=42)
Z, PG, DG = model.run_with_trajectories(T=50, weight_updates=1)

# Plot initial network state
initial_model = SocialNetworkMarkovChain(N=8, beta=50.0, sigma=0.1, seed=42)
initial_model.plot_network(title="Initial Network (t=0)")

# Plot opinion trajectories
model.plot_opinion_trajectories(Z)

# Plot polarization/disagreement
model.plot_polarization_disagreement(PG, DG)

# Plot final network
model.plot_network(title=f"Network After {len(PG)-1} Steps")