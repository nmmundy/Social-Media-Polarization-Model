import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

class OpinionNetwork:
    def __init__(self, N, beta=2.0, sigma=0.1, seed=None):
        self.N = N
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

        self.z = self.rng.uniform(0, 1, size=N)

        W = self.rng.uniform(0, 1, size=(N, N))
        np.fill_diagonal(W, 0)
        self.W = W / W.sum(axis=1, keepdims=True)

    def opinion_update(self):
        new_z = np.zeros_like(self.z)
        for i in range(self.N):
            weighted_sum = np.sum(self.W[i] * self.z)
            tilde_z = (self.z[i] + weighted_sum) / (1 + np.sum(self.W[i]))
            p = np.exp(-self.beta * abs(tilde_z - self.z[i]))
            new_z[i] = tilde_z if self.rng.random() < p else self.z[i]
        self.z = new_z

    def edge_weight_update(self):
        new_W = np.zeros_like(self.W)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    mu = 1 - abs(self.z[i] - self.z[j])
                    w_ij = self.rng.normal(mu, self.sigma)
                    new_W[i, j] = max(0, w_ij)
            if new_W[i].sum() > 0:
                new_W[i] /= new_W[i].sum()
        self.W = new_W

    def step(self):
        self.opinion_update()
        self.edge_weight_update()

    def run(self, T=10):
        opinion_history = [self.z.copy()]
        weight_history = [self.W.copy()]
        for _ in range(T):
            self.step()
            opinion_history.append(self.z.copy())
            weight_history.append(self.W.copy())
        return np.array(opinion_history), weight_history


def animate_combined_side_by_side(opinion_history, weight_history, interval=500):
    N = opinion_history.shape[1]
    T = opinion_history.shape[0]
    G = nx.complete_graph(N)
    pos = nx.circular_layout(G)

    # Create side-by-side figure
    fig, (ax_net, ax_traj) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
    colors = plt.cm.coolwarm(np.linspace(0, 1, N))

    # Prepare the trajectory plot
    lines = []
    for i in range(N):
        (line,) = ax_traj.plot([], [], color=colors[i], label=f"Node {i}")
        lines.append(line)
    ax_traj.set_xlim(0, T-1)
    ax_traj.set_ylim(0, 1)
    ax_traj.set_xlabel("Time Step")
    ax_traj.set_ylabel("Opinion")
    ax_traj.legend(loc="upper right")
    ax_traj.grid(True, linestyle='--', alpha=0.5)

    def update(frame):
        ax_net.clear()
        opinions = opinion_history[frame]
        weights = weight_history[frame]

        node_colors = plt.cm.coolwarm(opinions)
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edge_weights = [weights[i, j] for i, j in edges]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax_net)
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                               width=[3 * w for w in edge_weights],
                               edge_color='gray', alpha=0.7, ax=ax_net)
        nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(N)}, ax=ax_net)
        ax_net.set_title(f"Network at Time {frame}")
        ax_net.axis('off')

        # Update opinion trajectories
        for i in range(N):
            lines[i].set_data(range(frame + 1), opinion_history[:frame + 1, i])

    ani = FuncAnimation(fig, update, frames=T, interval=interval, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani


# Example usage:
if __name__ == "__main__":
    model = OpinionNetwork(N=6, beta=2.0, sigma=0.05, seed=42)
    opinion_history, weight_history = model.run(T=30)
    animate_combined_side_by_side(opinion_history, weight_history)
