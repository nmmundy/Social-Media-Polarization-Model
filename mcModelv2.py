import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
# import csv # Commented out as it's not strictly needed for heatmap generation, uncomment if you need it.
from matplotlib.colors import LinearSegmentedColormap


# --- MarkovChainPolarizationModel Class (as provided by you) ---
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
        self.w = w / w.sum(axis=1, keepdims=True)

    def opinionUpdate(self):
        z_prime = np.zeros_like(self.z)
        z_prime[0] = 0
        z_prime[1] = 1

        for t in range(2, self.n):
            ## Below two lines represent Equation 4 in the overleaf
            weighted_opinions = np.sum(self.w[t] * self.z)
            z_i = (self.z[t] + weighted_opinions) / (1 + np.sum(self.w[t]))
            ## Next line is Equation 5, the probabiliy
            p_i = np.exp(-self.eta * abs(z_i - self.z[t]))

            if (self.rng.random() < p_i):
                z_prime[t] = z_i

        self.z = z_prime

    ## Revised edge weights now company has influence
    ## self, platform input, target opinion zPrime, gamma strengh of company bias (0,1)
    ## concentration sets how tight the sampled weights are around mu
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
                        # print("Compay bias", companyBias) # Commented for heatmap generation verbosity
                        socialAffinity = np.clip(socialAffinity, 0, 1)
                        mu = 0.5 * companyBias * socialAffinity
                        # print("mu", mu) # Commented for heatmap generation verbosity
                    else:
                        mu = socialAffinity
                        # print("mu", mu) # Commented for heatmap generation verbosity
                    mu = np.clip(mu, 0.001, .999)

                    alpha = (mu) * concentration
                    beta_ = (1 - mu) * concentration

                    w_ij = np.random.beta(alpha, beta_)
                    '''
                    # Uncomment this block if you need to log weights to CSV during heatmap generation
                    with open('weightsV2.csv', 'a', newline='') as file:
                        fieldname = ["weight", "opinion", "companyBias", "mu", "alpha", "beta"]
                        writer = csv.DictWriter(file, fieldnames=fieldname)
                        #writer.writeheader()
                        writer.writerow({"weight": w_ij,
                                         "opinion": socialAffinity,
                                         "companyBias": companyBias,
                                         "mu": mu,
                                         "alpha": alpha,
                                         "beta": beta_
                                         })
                    '''
                    w_new[i, j] = w_ij

                # Small weights are set to 0 to simplify the graph
                if w_new[i, j] < .1:
                    w_new[i, j] = 0

            # Normalize weights for each node (row-wise sum to 1)
            row_sum = np.sum(w_new[i])
            if row_sum > 0:  # Avoid division by zero for isolated nodes
                w_new[i] = w_new[i] / row_sum

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

    def timeStep(self):
        self.opinionUpdate()
        self.edgeWeightUpdate()

    def runModel(self, t=30):
        # opinions = [self.z.copy()] # Only store final state for heatmap
        # weights = [self.w.copy()] # Only store final state for heatmap

        for _ in range(t):
            self.timeStep()
            # opinions.append(self.z.copy())
            # weights.append(self.w.copy())

        # return np.array(opinions), weights # Only return the final state
        return self.z.copy(), self.w.copy()  # Return final opinions and weights


# --- Revised Visualization Function (as provided by you, for animation) ---
def visualization(opinions, weights, interval=250):
    n = opinions.shape[1]
    t = opinions.shape[0]

    # Create the graph object, which will be updated
    Graph = nx.Graph()
    Graph.add_nodes_from(range(n))

    # We use a stable, initial layout for the y-coordinates
    # initial_y_pos = {i: 2 * (i / (n - 1)) - 1 for i in range(n)} # Not used in final spring_layout call

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
    network.set_ylim(-1.1, 1.1)  # Set limits to prevent plot from resizing

    # --- The Animation Update Function ---
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

        # Calculate the layout using nx.spring_layout
        # pos=None means it will generate an initial random layout or use the previous one if available
        # k: ideal distance between nodes.
        # iterations: one step of the layout algorithm per frame for smooth movement.
        # Fixed seed for spring_layout can make animations consistent if needed
        # We can pass the previous position `pos` to `spring_layout` for smoother transitions
        # However, the user's original code does not pass `pos` as an argument to spring_layout's `pos`
        # and re-calculates it each time, which can lead to jitter if iterations is too low or k is off.
        # For smooth animations, `pos` should be updated incrementally.
        # For now, keeping it as the user had it.
        pos = nx.spring_layout(Graph, pos=None, weight='weight', k=0.5, iterations=5, scale=1)

        # Set the x-position to opinion value and the y-position to a constant for a clearer visualization
        for node in Graph.nodes():
            pos[node][0] = curr_opinions[node]
            # If you want a more stable y-axis in the network, uncomment the line below.
            # Otherwise, spring_layout will determine both x and y freely.
            # pos[node][1] = initial_y_pos[node]

        node_colors = plt.cm.coolwarm(curr_opinions)

        # Draw the network with the new layout
        edge_widths = [d['weight'] * 5 for u, v, d in Graph.edges(data=True)]

        nx.draw_networkx_nodes(Graph, pos, node_color=node_colors, node_size=400, ax=network)
        nx.draw_networkx_edges(Graph, pos, width=edge_widths, edge_color='gray', alpha=0.7, ax=network)
        nx.draw_networkx_labels(Graph, pos, {i: str(i) for i in range(n)}, ax=network)

        network.set_title(f"Time Step: {frame}")
        network.axis('off')  # Hide axes for cleaner network visualization

        # Set stable limits to prevent the plot from resizing
        network.set_xlim(-0.1, 1.1)
        network.set_ylim(-1.1, 1.1)

        # Update opinion trajectories
        for i in range(n):
            lines[i].set_data(range(frame + 1), opinions[:frame + 1, i])

        # Return all artists that have been modified for blitting
        # For blit=False, this return value is not strictly necessary but good practice.
        return lines + network.collections + network.patches + network.texts

    # --- Create the Animation ---
    ani = FuncAnimation(fig, update, frames=t, interval=interval, repeat=False, blit=False)
    plt.tight_layout()
    plt.show()
    return ani


# --- Function to calculate Polarization (Spectral Gap) for Heatmap ---
def calculate_polarization_for_heatmap(zprime_val, zeta_val):
    """
    Calculates the 'polarization' value (spectral gap) using the MarkovChainPolarizationModel
    for given zprime and zeta values.

    Args:
        zprime_val (float): The target opinion of the company.
        zeta_val (float): The strength of company bias.

    Returns:
        float: The calculated spectral gap, representing polarization.
    """
    # Instantiate the model with the given zprime and zeta
    # Use consistent parameters for n, eta, beta, alpha, sigma, seed for the heatmap calculation.
    # These parameters define the *environment* in which polarization is measured.
    model_instance = MarkovChainPolarizationModel(
        n=15,  # Number of nodes
        eta=0.5,  # Sensitivity to opinion updates
        beta=3.0,  # Beta distribution parameter (from your example)
        alpha=2.0,  # Beta distribution parameter (from your example)
        sigma=0.1,  # Standard deviation for edge weights (from your example)
        seed=42,  # Fixed seed for reproducibility across heatmap calculations
        zprime=zprime_val,
        zeta=zeta_val
    )

    # Run the model for a few time steps to allow opinions/weights to stabilize.
    # The number of steps here might need tuning based on how quickly your model converges.
    run_steps = 50
    final_opinions, final_weights = model_instance.runModel(t=run_steps)

    # Get the spectral gap from the final state of the model
    _, _, _, spectral_gap = model_instance.laplacianSpectralGap()

    return spectral_gap


# --- Heatmap Generation Function ---
def generate_polarization_heatmap(zprime_min, zprime_max, zeta_min, zeta_max, resolution):
    """
    Generates and displays a heatmap of polarization values (spectral gap).

    Args:
        zprime_min (float): Minimum value for zprime.
        zprime_max (float): Maximum value for zprime.
        zeta_min (float): Minimum value for zeta.
        zeta_max (float): Maximum value for zeta.
        resolution (int): Number of steps for both zprime and zeta axes.
                          Higher resolution means more detail but longer computation time.
    """

    # Create arrays for zprime and zeta values
    zprime_values = np.linspace(zprime_min, zprime_max, resolution)
    zeta_values = np.linspace(zeta_min, zeta_max, resolution)

    # Initialize a 2D array to store polarization values
    polarization_data = np.zeros((resolution, resolution))

    print(
        f"Generating heatmap for zprime from {zprime_min} to {zprime_max} and zeta from {zeta_min} to {zeta_max} with resolution {resolution}x{resolution}...")

    # Calculate polarization for each combination of zprime and zeta
    for i in range(resolution):
        # Progress indicator (optional, as calculation can be long for high resolution)
        if (i + 1) % (resolution // 10) == 0 or i == resolution - 1:
            print(f"Calculating row {i + 1}/{resolution}...")
        for j in range(resolution):
            current_zprime = zprime_values[j]  # zprime maps to x-axis
            current_zeta = zeta_values[i]  # zeta maps to y-axis (row index)
            polarization_data[i, j] = calculate_polarization_for_heatmap(current_zprime, current_zeta)

    # --- Plotting the Heatmap ---
    plt.figure(figsize=(10, 8))

    # Define a custom colormap that clearly shows varying polarization.
    # A diverging colormap like 'RdYlGn' or 'seismic' can be good if polarization
    # can be both positive and negative, or if a clear "neutral" value exists.
    # For spectral gap, often lower values mean more polarization (bad), higher values mean less (good).
    # So, we might want to map low values to a "red/bad" color and high values to a "green/good" color.
    # The current "blue, white, red" is also a good diverging map if center is neutral.
    # Let's use a standard diverging map or define one based on meaning of polarization.
    # Assuming lower spectral gap = higher polarization (as per your print statement "Polarizatoin smaller = more")
    # We want red for small gap (high polarization) and blue for large gap (low polarization)
    colors = ["red", "white", "blue"]  # From high polarization (low gap) to low polarization (high gap)
    cmap = LinearSegmentedColormap.from_list("polarization_cmap", colors, N=256)

    # Use imshow to create the heatmap
    # extent: [xmin, xmax, ymin, ymax] for correct axis labeling
    # origin='lower': Makes (0,0) the bottom-left corner, matching standard plots
    img = plt.imshow(polarization_data, cmap=cmap, origin='lower',
                     extent=[zprime_min, zprime_max, zeta_min, zeta_max],
                     aspect='auto')  # aspect='auto' adjusts to fit the figure size

    plt.colorbar(img, label='Spectral Gap (Polarization Value)')
    plt.xlabel("Z-prime (Target Opinion)")
    plt.ylabel("Zeta (Strength of Company Bias)")
    plt.title("Polarization (Spectral Gap) Heatmap for Z-prime and Zeta")
    plt.grid(True, linestyle='--', alpha=0.6)  # Add a subtle grid
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Parameters for Heatmap Generation ---
    z_prime_min = 0.0
    z_prime_max = 1.0
    zeta_min_val = 0.0
    zeta_max_val = 10.0
    resolution_val = 50  # Lower resolution for faster initial testing, increase for detail

    # Generate the heatmap
    generate_polarization_heatmap(z_prime_min, z_prime_max, zeta_min_val, zeta_max_val, resolution_val)

    # --- Example Usage of your MarkovChainPolarizationModel with visualization ---
    # You can uncomment the following lines to run and visualize a single simulation
    # model_animation = MarkovChainPolarizationModel(n=15, eta=.5, sigma=.05, seed=32, zprime=1, zeta=5)
    # opinions_animation, weights_animation = model_animation.runModel(t=100)
    # _, _, eigvals_animation, gap_animation = model_animation.laplacianSpectralGap()
    # print(f"\nExample Animation Run Spectral Gap (Polarization): {gap_animation} (smaller = more polarized)")
    # visualization(opinions_animation, weights_animation)
