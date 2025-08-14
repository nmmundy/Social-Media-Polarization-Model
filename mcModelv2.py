import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
# import csv # Commented out as it's not strictly needed for heatmap generation, uncomment if you need it.
from matplotlib.colors import LinearSegmentedColormap


# --- MarkovChainPolarizationModel Class ---
class MarkovChainPolarizationModel:
    def __init__(self, n, eta=1., beta=3.0, alpha=2.0, sigma=0.1, seed=None, zprime=1, zeta=1):
        self.n = n  # number of nodes
        self.eta = eta  # sensitivity to opinion updates parameter
        self.sigma = sigma  # standard deviation for edge weights
        self.rng = np.random.default_rng(seed)  # random number generator
        self.beta = beta  # parameter for the beta distribution
        self.alpha = alpha  # parameter for beta distribution
        self.z = self.rng.uniform(0, 1, size=n)  # opinions of each node at time step t
        self.z[n - 1] = 0
        self.z[n - 2] = 1
        self.zprime = zprime  # target opinion of company
        self.zeta = zeta  # strength of company bias

        w = self.rng.uniform(0, 1, size=(n, n))  # weights of edges at current time
        np.fill_diagonal(w, 0)
        # Ensure that rows sum to 1 to represent probabilities correctly,
        # handling cases where sum might be zero (e.g., small n or initial weights)
        row_sums = w.sum(axis=1, keepdims=True)
        # Avoid division by zero: if row_sum is 0, keep row as 0
        self.w = np.divide(w, row_sums, out=np.zeros_like(w), where=row_sums != 0)

    def opinionUpdate(self):
        z_prime = np.zeros_like(self.z)
        z_prime[0] = 0
        z_prime[1] = 1

        for t in range(2, self.n):
            ## Below two lines represent Equation 4 in the overleaf
            # Sum of weights from node t to all other nodes (row t of W)
            sum_w_t = np.sum(self.w[t])

            # Avoid division by zero if node t has no outgoing connections
            if sum_w_t == 0:
                z_i = self.z[t]  # Opinion remains unchanged if no connections
            else:
                weighted_opinions = np.sum(self.w[t] * self.z)
                z_i = (self.z[t] + weighted_opinions) / (1 + sum_w_t)

            ## Next line is Equation 5, the probabiliy
            p_i = np.exp(-self.eta * abs(z_i - self.z[t]))

            if (self.rng.random() < p_i):
                z_prime[t] = z_i
            else:
                z_prime[t] = self.z[t]  # Opinion does not update if condition not met

        self.z = z_prime

    def edgeWeightUpdate(self, platformInfluence=True, delta=.5):
        zPrime = self.zprime
        zeta = self.zeta
        concentration = 40
        w_new = np.zeros_like(self.w)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    userOpinion = abs(self.z[i] - self.z[j])
                    socialAffinity = 1 - delta * userOpinion
                    if platformInfluence:
                        companyBias = np.exp(-zeta * abs((self.z[i] + self.z[j]) / 2 - zPrime))
                        socialAffinity = np.clip(socialAffinity, 0, 1)
                        mu = 0.5 * companyBias * socialAffinity
                    else:
                        mu = socialAffinity
                    mu = np.clip(mu, 0.001, .999)  # Ensure mu is within (0,1) for beta distribution

                    alpha = (mu) * concentration
                    beta_ = (1 - mu) * concentration

                    # Sample from Beta distribution
                    w_ij = self.rng.beta(alpha, beta_)

                    w_new[i, j] = w_ij

                # Small weights are set to 0 to simplify the graph
                if w_new[i, j] < .1:
                    w_new[i, j] = 0

            # Normalize weights for each node (row-wise sum to 1)
            row_sum = np.sum(w_new[i])
            if row_sum > 0:  # Avoid division by zero for isolated nodes
                w_new[i] = w_new[i] / row_sum
            else:
                # If a row sums to 0, it means the node is isolated,
                # so its outgoing weights remain 0.
                pass

        self.w = w_new

    def laplacianSpectralGap(self):
        # Degree matrix
        D = np.diag(self.w.sum(axis=1))

        # Laplacian
        L = D - self.w

        # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warnings for isolated nodes
            D_inv_sqrt = np.diag(1.0 / np.sqrt(self.w.sum(axis=1)))
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0  # Set inf to 0 for isolated nodes
            D_inv_sqrt[np.isnan(D_inv_sqrt)] = 0.0  # Set NaN to 0 if there are NaN values

        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Eigenvalues
        # Use eigvalsh for symmetric matrices for better numerical stability
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)  # Sort eigenvalues in ascending order

        # Spectral gap = λ_2 (second-smallest eigenvalue)
        # It's a measure of graph connectivity. For a connected graph, λ_1 = 0.
        # If the graph has multiple connected components, λ_2 = 0.
        if len(eigenvalues) >= 2:
            # Check for near-zero eigenvalues for disconnected components
            # If the smallest eigenvalue is not close to zero, there might be an issue
            # For a connected graph, the smallest eigenvalue should be 0 or very close to 0
            if np.isclose(eigenvalues[0], 0):
                spectral_gap = eigenvalues[1]
            else:
                # If smallest eigenvalue is not 0, it indicates an issue or a non-standard graph
                # For polarization, a non-zero smallest eigenvalue might still be relevant
                spectral_gap = eigenvalues[1]
        else:
            spectral_gap = 0  # No spectral gap if fewer than 2 eigenvalues (e.g., 1 node or disconnected)

        return L, L_norm, eigenvalues, spectral_gap

    # --- NEW METHOD: Calculate custom polarization metric ---
    def calculate_opinion_alignment_polarization(self):
        """
        Calculates the custom polarization metric: Sum over i,j (z_i - 1/2)(z_j - 1/2)(w_ij).

        A more negative value indicates higher polarization (disagreement across strong links).
        A more positive value indicates strong homophily/consensus.
        """
        polarization_sum = 0.0
        for i in range(self.n):
            for j in range(self.n):
                # Ensure we only sum non-zero weights or skip self-loops if w_ii is meant to be 0
                if self.w[i, j] > 0 and i != j:  # Explicitly exclude self-loops as per standard graph theory
                    term_i = self.z[i] - 0.5
                    term_j = self.z[j] - 0.5
                    polarization_sum += term_i * term_j * self.w[i, j]
        return polarization_sum

    def timeStep(self):
        self.opinionUpdate()
        self.edgeWeightUpdate()

    def runModel(self, t=30):
        # We only need the final state for the heatmap calculation
        for _ in range(t):
            self.timeStep()

        return self.z.copy(), self.w.copy()  # Return final opinions and weights


# --- Revised Visualization Function (for animation, unchanged) ---
def visualization(opinions, weights, interval=250):
    n = opinions.shape[1]
    t = opinions.shape[0]

    # Create the graph object, which will be updated
    Graph = nx.Graph()
    Graph.add_nodes_from(range(n))

    fig, (network, opinion_traj) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))

    # --- Opinion Trajectory Setup ---
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

    # --- The Animation Update Function ---
    def update(frame):
        network.clear()

        curr_opinions = opinions[frame]
        curr_weights_matrix = weights[frame]

        Graph.clear_edges()
        for i in range(n):
            for j in range(i + 1, n):
                weight = curr_weights_matrix[i, j]
                if weight > 0:
                    Graph.add_edge(i, j, weight=weight)

        pos = nx.spring_layout(Graph, pos=None, weight='weight', k=0.5, iterations=5, scale=1)

        for node in Graph.nodes():
            pos[node][0] = curr_opinions[node]

        node_colors = plt.cm.coolwarm(curr_opinions)

        edge_widths = [d['weight'] * 5 for u, v, d in Graph.edges(data=True)]

        nx.draw_networkx_nodes(Graph, pos, node_color=node_colors, node_size=400, ax=network)
        nx.draw_networkx_edges(Graph, pos, width=edge_widths, edge_color='gray', alpha=0.7, ax=network)
        nx.draw_networkx_labels(Graph, pos, {i: str(i) for i in range(n)}, ax=network)

        network.set_title(f"Time Step: {frame}")
        network.axis('off')

        network.set_xlim(-0.1, 1.1)
        network.set_ylim(-1.1, 1.1)

        for i in range(n):
            lines[i].set_data(range(frame + 1), opinions[:frame + 1, i])

        return lines + network.collections + network.patches + network.texts

    ani = FuncAnimation(fig, update, frames=t, interval=interval, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


# --- Function to calculate Polarization (Custom Metric) for Heatmap ---
def calculate_polarization_for_heatmap(zprime_val, zeta_val):
    """
    Calculates the 'polarization' value using the MarkovChainPolarizationModel
    for given zprime and zeta values, using the custom sum metric.

    Args:
        zprime_val (float): The target opinion of the company.
        zeta_val (float): The strength of company bias.

    Returns:
        float: The calculated custom polarization value.
    """
    model_instance = MarkovChainPolarizationModel(
        n=15,  # Number of nodes
        eta=0.5,  # Sensitivity to opinion updates
        beta=3.0,  # Beta distribution parameter
        alpha=2.0,  # Beta distribution parameter
        sigma=0.1,  # Standard deviation for edge weights
        seed=42,  # Fixed seed for reproducibility across heatmap calculations
        zprime=zprime_val,
        zeta=zeta_val
    )

    run_steps = 50  # Number of steps to run the model before calculating polarization
    model_instance.runModel(t=run_steps)  # Run the model to reach a stable state

    # Get the custom polarization from the final state of the model
    custom_polarization = model_instance.calculate_opinion_alignment_polarization()

    return custom_polarization


# --- Heatmap Generation Function ---
def generate_polarization_heatmap(zprime_min, zprime_max, zeta_min, zeta_max, resolution):
    """
    Generates and displays a heatmap of custom polarization values.

    Args:
        zprime_min (float): Minimum value for zprime.
        zprime_max (float): Maximum value for zprime.
        zeta_min (float): Minimum value for zeta.
        zeta_max (float): Maximum value for zeta.
        resolution (int): Number of steps for both zprime and zeta axes.
                          Higher resolution means more detail but longer computation time.
    """

    zprime_values = np.linspace(zprime_min, zprime_max, resolution)

    # Reverting to linear scale for zeta
    zeta_values = np.linspace(zeta_min, zeta_max, resolution)

    polarization_data = np.zeros((resolution, resolution))

    print(
        f"Generating heatmap for zprime from {zprime_min} to {zprime_max} and zeta from {zeta_min} to {zeta_max} (linear scale) with resolution {resolution}x{resolution}...")

    for i in range(resolution):
        if (i + 1) % (resolution // 10) == 0 or i == resolution - 1:
            print(f"Calculating row {i + 1}/{resolution}...")
        for j in range(resolution):
            current_zprime = zprime_values[j]
            current_zeta = zeta_values[i]
            polarization_data[i, j] = calculate_polarization_for_heatmap(current_zprime, current_zeta)

    # --- Plotting the Heatmap ---
    plt.figure(figsize=(10, 8))

    # For this new metric:
    #   Negative values: higher polarization (disagreement across links)
    #   Positive values: lower polarization (homophily/agreement across links)
    # Using 'seismic' colormap: blue for negative, white for zero, red for positive.
    # So, blue will be higher polarization, red will be lower polarization (more alignment).
    cmap = plt.cm.seismic  # Blue (negative) -> White (zero) -> Red (positive)

    # You can reverse it if you want red for high polarization (negative values)
    # cmap = plt.cm.seismic_r # Red (negative) -> White (zero) -> Blue (positive)

    # To ensure the heatmap spans the full range of values and centers around 0,
    # find min/max and use vmin/vmax with a symmetric color scale.
    max_abs_val = np.max(np.abs(polarization_data))

    img = plt.imshow(polarization_data, cmap=cmap, origin='lower',
                     extent=[zprime_min, zprime_max, zeta_min, zeta_max],
                     aspect='auto',
                     vmin=-max_abs_val, vmax=max_abs_val)  # Symmetrically center colormap around zero

    plt.colorbar(img, label='Opinion Alignment Polarization Sum')
    plt.xlabel("Z-prime (Company Target Opinion)")
    # Reverting to linear scale label for zeta
    plt.ylabel("Zeta (Strength of Company Bias) - Linear Scale")
    # Removed plt.yscale('log')
    plt.title("Opinion Alignment Polarization Heatmap")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Parameters for Heatmap Generation ---
    z_prime_min = 0.0
    z_prime_max = 1.0
    zeta_min_val = 0.0  # Reverted to 0.0 for linear scale
    zeta_max_val = 100.0
    resolution_val = 25  # Lower resolution for faster initial testing, increase for detail

    # Generate the heatmap
    generate_polarization_heatmap(z_prime_min, z_prime_max, zeta_min_val, zeta_max_val, resolution_val)

    # --- Example Usage of your MarkovChainPolarizationModel with visualization ---
    # You can uncomment the following lines to run and visualize a single simulation
    # model_animation = MarkovChainPolarizationModel(n=15, eta=.5, sigma=.05, seed=32, zprime=1, zeta=5)
    # opinions_animation, weights_animation = model_animation.runModel(t=100)
    # _, _, eigvals_animation, gap_animation = model_animation.laplacianSpectralGap()
    # print(f"\nExample Animation Run Spectral Gap (Polarization): {gap_animation} (smaller = more polarized)")
    # visualization(opinions_animation, weights_animation)
