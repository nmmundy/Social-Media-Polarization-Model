import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Note: The following four functions describe a static equilibrium model (Friedkin-Johnsen)
# and are not used in the dynamic simulation loop below. They are kept here for reference.
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

def FreidkinJohnsonEquilibruim(G, s):
    L = nx.laplacian_matrix(G).astype(float)
    I = identity(L.shape[0])
    return spsolve(I + L, s)

def disagreement(G, z):
    return sum((w) * (z[u] - v)**2 for u, v, w in G.edges(data='weight'))

def opinion(z):
    z_spread = z - np.mean(z)
    return np.dot(z_spread, z_spread)

def nodeStress(G, s):
    z = FreidkinJohnsonEquilibruim(G, s)
    return disagreement(G, z) + opinion(z), z


# --- Functions for the Dynamic Simulation ---

def update_all_opinions(G: nx.Graph, z: np.ndarray, beta: float) -> np.ndarray:
    """
    Performs a synchronous opinion update for all nodes in the network.

    Args:
        G (nx.Graph): The network with 'weight' attributes on edges.
        z (np.ndarray): The current array of opinions for all nodes.
        beta (float): The sensitivity parameter for opinion change.

    Returns:
        np.ndarray: The updated array of opinions.
    """
    new_z = z.copy()
    n = len(z)
    for i in range(n):
        # Calculate the weighted sum of neighbor opinions
        weighted_sum_opinions = sum(G[i][j]['weight'] * z[j] for j in G.neighbors(i))
        # The sum of weights for outgoing edges from a node is 1.0
        denominator = 1 + sum(G[i][j]['weight'] for j in G.neighbors(i))

        # Calculate the proposed new opinion
        proposed_z = (z[i] + weighted_sum_opinions) / denominator

        # Acceptance probability based on resistance to change
        p_accept = np.exp(-beta * abs(proposed_z - z[i]))

        # Bernoulli trial: update opinion with probability p_accept
        if random.random() < p_accept:
            new_z[i] = proposed_z
    return new_z


def update_all_weights(G: nx.Graph, z: np.ndarray, sigma: float) -> nx.Graph:
    """
    Performs a synchronous weight update for all edges based on homophily.

    Args:
        G (nx.Graph): The current network.
        z (np.ndarray): The current array of opinions.
        sigma (float): The standard deviation for the weight update.

    Returns:
        nx.Graph: The network with updated edge weights.
    """
    n = len(z)
    for i in range(n):
        new_weights = {}
        # Calculate new weights based on opinion similarity (homophily)
        for j in G.neighbors(i):
            mean = 1 - abs(z[i] - z[j])
            # Draw from a normal distribution and clip to [0, 1]
            new_weight = np.clip(random.gauss(mean, sigma), 0, 1)
            new_weights[j] = new_weight

        # Normalize the new weights for the current node
        total_weight = sum(new_weights.values())
        for j, w in new_weights.items():
            if total_weight > 0:
                G[i][j]['weight'] = w / total_weight
            else:
                # Fallback for the rare case where all weights are zero
                G[i][j]['weight'] = 1.0 / (n - 1)
    return G


# --- Main Simulation Block ---

if __name__ == "__main__":
    # 1. DEFINE PARAMETERS
    N_USERS = 20         # Number of users (nodes)
    TIME_STEPS = 150     # Number of time steps for the simulation
    BETA = 5.0           # Controls resistance to opinion change (higher means more resistance)
    SIGMA = 0.4          # Controls randomness of weight updates (higher means more random)

    print(f"🚀 Starting simulation with {N_USERS} users for {TIME_STEPS} steps.")

    # 2. INITIALIZE THE NETWORK AND OPINIONS
    # Create a complete graph where everyone is connected to everyone
    G = nx.complete_graph(N_USERS)
    # Assign initial random opinions to each user
    z = np.random.rand(N_USERS)
    # Assign initial random (and normalized) weights to each edge
    for i in G.nodes():
        # Generate random weights for all outgoing edges
        weights = {j: random.random() for j in G.neighbors(i)}
        total = sum(weights.values())
        # Normalize weights so they sum to 1 for each node
        for j, w in weights.items():
            G[i][j]['weight'] = w / total if total > 0 else 0

    # 3. RUN THE SIMULATION
    opinion_history = [z.copy()]  # Store the initial state

    for t in range(TIME_STEPS):
        # In each step, first update all opinions, then all weights
        z = update_all_opinions(G, z, BETA)
        G = update_all_weights(G, z, SIGMA)

        # Store the opinions at this time step
        opinion_history.append(z.copy())
        if (t + 1) % 25 == 0:
            print(f"   ...Completed step {t+1}/{TIME_STEPS}")

    print("✅ Simulation complete.")

    # 4. PLOT THE RESULTS
    history_array = np.array(opinion_history)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(history_array)
    plt.title('Opinion Dynamics Over Time', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Opinion', fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.show()