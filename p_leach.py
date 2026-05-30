import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================================================
# 1. SIMULATION PARAMETERS
# =========================================================
AREA_X = 600
AREA_Y = 600
NUM_NODES = 100          # Use 100 first for debugging
NUM_ROUNDS = 180
INITIAL_ENERGY = 5.0     # Joules

# Sink / Base Station
SINK_X = 150
SINK_Y = 50

# Packet
PACKET_SIZE_BYTES = 256
PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8

# Clustering
NUM_CLUSTERS = 4

# Radio energy model constants
E_ELEC = 50e-9
E_FS = 10e-12
E_MP = 0.0013e-12
E_DA = 5e-9

# Threshold distance
D0 = math.sqrt(E_FS / E_MP)

# CH replacement threshold
CH_ENERGY_THRESHOLD = 0.2 * INITIAL_ENERGY

plt.rcParams["figure.figsize"] = (8, 8)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distance_to_sink(row, sink_x=SINK_X, sink_y=SINK_Y):
    return euclidean_distance(row["x"], row["y"], sink_x, sink_y)


# =========================================================
# 3. ENERGY MODEL
# =========================================================
def tx_energy(bits, distance):
    if distance < D0:
        return bits * E_ELEC + bits * E_FS * (distance ** 2)
    else:
        return bits * E_ELEC + bits * E_MP * (distance ** 4)


def rx_energy(bits):
    return bits * E_ELEC


def da_energy(bits):
    return bits * E_DA


# =========================================================
# 4. NETWORK INITIALIZATION
# =========================================================
def initialize_nodes(num_nodes=NUM_NODES, area_x=AREA_X, area_y=AREA_Y, initial_energy=INITIAL_ENERGY):
    nodes = pd.DataFrame({
        "id": np.arange(num_nodes),
        "x": np.random.uniform(0, area_x, num_nodes),
        "y": np.random.uniform(0, area_y, num_nodes),
        "energy": np.full(num_nodes, initial_energy, dtype=float),
        "alive": np.ones(num_nodes, dtype=bool),
        "cluster_id": np.full(num_nodes, -1, dtype=int),
        "is_CH": np.zeros(num_nodes, dtype=bool),
        "is_leader": np.zeros(num_nodes, dtype=bool),
        "times_selected_as_CH": np.zeros(num_nodes, dtype=int),
        "dist_to_sink": np.zeros(num_nodes, dtype=float)
    })
    return nodes


# =========================================================
# 5. CLUSTER ASSIGNMENT
# Fixed 4-region clustering for easy first implementation
# =========================================================
def assign_clusters(nodes):
    nodes = nodes.copy()

    for i in nodes.index:
        x = nodes.at[i, "x"]
        y = nodes.at[i, "y"]

        if x <= AREA_X / 2 and y <= AREA_Y / 2:
            cid = 0
        elif x <= AREA_X / 2 and y > AREA_Y / 2:
            cid = 1
        elif x > AREA_X / 2 and y <= AREA_Y / 2:
            cid = 2
        else:
            cid = 3

        nodes.at[i, "cluster_id"] = cid

    return nodes


# =========================================================
# 6. CLUSTER HEAD SELECTION
# Rule-based: highest residual energy in each cluster
# AI can replace this function later
# =========================================================
def select_cluster_heads(nodes):
    nodes = nodes.copy()
    nodes["is_CH"] = False
    nodes["is_leader"] = False

    ch_indices = []

    for cid in sorted(nodes["cluster_id"].unique()):
        cluster_nodes = nodes[(nodes["cluster_id"] == cid) & (nodes["alive"] == True)]

        if len(cluster_nodes) == 0:
            continue

        max_energy = cluster_nodes["energy"].max()
        candidates = cluster_nodes[cluster_nodes["energy"] == max_energy]

        # Tie-break: choose the one nearest to sink
        chosen_idx = candidates["dist_to_sink"].idxmin()

        nodes.at[chosen_idx, "is_CH"] = True
        nodes.at[chosen_idx, "times_selected_as_CH"] += 1
        ch_indices.append(chosen_idx)

    return nodes, ch_indices


# =========================================================
# 7. LEADER SELECTION
# Paper uses nearest distance to BS and high energy
# First implementation: nearest CH to sink
# =========================================================
def select_leader(nodes, ch_indices):
    nodes = nodes.copy()

    if len(ch_indices) == 0:
        return nodes, None

    ch_nodes = nodes.loc[ch_indices]
    leader_idx = ch_nodes["dist_to_sink"].idxmin()
    nodes.at[leader_idx, "is_leader"] = True

    return nodes, leader_idx


# =========================================================
# 8. CHAIN FORMATION AMONG CLUSTER HEADS
# Nearest-neighbor chain ending at leader
# =========================================================
def build_ch_chain(nodes, ch_indices, leader_idx):
    if leader_idx is None:
        return []

    if len(ch_indices) == 1:
        return [leader_idx]

    remaining = [idx for idx in ch_indices if idx != leader_idx]
    chain = []

    leader_x = nodes.at[leader_idx, "x"]
    leader_y = nodes.at[leader_idx, "y"]

    # Start from farthest CH from leader
    start_idx = max(
        remaining,
        key=lambda idx: euclidean_distance(nodes.at[idx, "x"], nodes.at[idx, "y"], leader_x, leader_y)
    )

    chain.append(start_idx)
    remaining.remove(start_idx)

    current = start_idx
    while remaining:
        next_idx = min(
            remaining,
            key=lambda idx: euclidean_distance(
                nodes.at[current, "x"], nodes.at[current, "y"],
                nodes.at[idx, "x"], nodes.at[idx, "y"]
            )
        )
        chain.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx

    chain.append(leader_idx)
    return chain


# =========================================================
# 9. INTRA-CLUSTER TRANSMISSION
# Normal nodes -> Cluster Head
# =========================================================
def intra_cluster_transmission(nodes, packet_bits=PACKET_SIZE_BITS):
    nodes = nodes.copy()

    for cid in sorted(nodes["cluster_id"].unique()):
        cluster_nodes = nodes[(nodes["cluster_id"] == cid) & (nodes["alive"] == True)]
        ch_nodes = cluster_nodes[cluster_nodes["is_CH"] == True]

        if len(ch_nodes) == 0:
            continue

        ch_idx = ch_nodes.index[0]
        ch_x, ch_y = nodes.at[ch_idx, "x"], nodes.at[ch_idx, "y"]

        for idx in cluster_nodes.index:
            if idx == ch_idx:
                continue

            if not nodes.at[idx, "alive"]:
                continue

            d = euclidean_distance(nodes.at[idx, "x"], nodes.at[idx, "y"], ch_x, ch_y)

            e_tx = tx_energy(packet_bits, d)
            e_rx = rx_energy(packet_bits)
            e_da = da_energy(packet_bits)

            nodes.at[idx, "energy"] -= e_tx
            nodes.at[ch_idx, "energy"] -= (e_rx + e_da)

    return nodes


# =========================================================
# 10. INTER-CLUSTER TRANSMISSION
# CH -> CH along chain, then leader -> sink
# =========================================================
def inter_cluster_transmission(nodes, chain, packet_bits=PACKET_SIZE_BITS):
    nodes = nodes.copy()

    if len(chain) == 0:
        return nodes

    # CH to CH forwarding
    for i in range(len(chain) - 1):
        src = chain[i]
        dst = chain[i + 1]

        d = euclidean_distance(
            nodes.at[src, "x"], nodes.at[src, "y"],
            nodes.at[dst, "x"], nodes.at[dst, "y"]
        )

        nodes.at[src, "energy"] -= tx_energy(packet_bits, d)
        nodes.at[dst, "energy"] -= (rx_energy(packet_bits) + da_energy(packet_bits))

    # Leader to sink
    leader = chain[-1]
    d_sink = euclidean_distance(nodes.at[leader, "x"], nodes.at[leader, "y"], SINK_X, SINK_Y)
    nodes.at[leader, "energy"] -= tx_energy(packet_bits, d_sink)

    return nodes


# =========================================================
# 11. DEAD NODE UPDATE
# =========================================================
def update_dead_nodes(nodes):
    nodes = nodes.copy()
    nodes.loc[nodes["energy"] <= 0, "alive"] = False
    nodes.loc[nodes["energy"] < 0, "energy"] = 0
    return nodes


# =========================================================
# 12. OPTIONAL: WEAK CH REPLACEMENT CHECK
# Current rule-based version. Can be used later if needed.
# =========================================================
def replace_weak_cluster_heads(nodes, threshold=CH_ENERGY_THRESHOLD):
    nodes = nodes.copy()

    for cid in sorted(nodes["cluster_id"].unique()):
        cluster_nodes = nodes[(nodes["cluster_id"] == cid) & (nodes["alive"] == True)]
        current_ch = cluster_nodes[cluster_nodes["is_CH"] == True]

        if len(current_ch) == 0:
            continue

        ch_idx = current_ch.index[0]
        if nodes.at[ch_idx, "energy"] > threshold:
            continue

        nodes.at[ch_idx, "is_CH"] = False

        max_energy = cluster_nodes["energy"].max()
        candidates = cluster_nodes[cluster_nodes["energy"] == max_energy]
        new_ch_idx = candidates["dist_to_sink"].idxmin()

        nodes.at[new_ch_idx, "is_CH"] = True
        nodes.at[new_ch_idx, "times_selected_as_CH"] += 1

    return nodes


# =========================================================
# 13. FEATURE COLLECTION FOR FUTURE AI
# Saves node info each round
# =========================================================
def collect_node_features(nodes, round_no):
    df = nodes.copy()
    df["round"] = round_no
    df["selected_as_CH"] = df["is_CH"].astype(int)

    return df[[
        "round", "id", "x", "y", "energy", "dist_to_sink",
        "cluster_id", "times_selected_as_CH", "selected_as_CH", "alive"
    ]]


# =========================================================
# 14. PLOTTING FUNCTIONS
# =========================================================
def plot_network(nodes, title="Initial Network"):
    plt.figure()
    plt.scatter(nodes["x"], nodes["y"], c="blue", label="Sensor Nodes")
    plt.scatter(SINK_X, SINK_Y, c="red", marker="*", s=200, label="Base Station")
    plt.xlim(0, AREA_X)
    plt.ylim(0, AREA_Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_clusters(nodes, title="Clusters"):
    plt.figure()
    colors = ["blue", "green", "orange", "purple", "brown", "cyan"]

    for cid in sorted(nodes["cluster_id"].unique()):
        cluster_nodes = nodes[nodes["cluster_id"] == cid]
        plt.scatter(cluster_nodes["x"], cluster_nodes["y"], label=f"Cluster {cid}", color=colors[cid % len(colors)])

    plt.scatter(SINK_X, SINK_Y, c="red", marker="*", s=200, label="Base Station")
    plt.xlim(0, AREA_X)
    plt.ylim(0, AREA_Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cluster_heads(nodes, title="Cluster Heads"):
    plt.figure()

    normal_nodes = nodes[nodes["is_CH"] == False]
    ch_nodes = nodes[nodes["is_CH"] == True]

    plt.scatter(normal_nodes["x"], normal_nodes["y"], c="lightblue", label="Normal Nodes")
    plt.scatter(ch_nodes["x"], ch_nodes["y"], c="black", s=120, label="Cluster Heads")
    plt.scatter(SINK_X, SINK_Y, c="red", marker="*", s=200, label="Base Station")

    plt.xlim(0, AREA_X)
    plt.ylim(0, AREA_Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_chain(nodes, chain, title="P-LEACH CH Chain"):
    plt.figure()

    normal_nodes = nodes[nodes["is_CH"] == False]
    ch_nodes = nodes[nodes["is_CH"] == True]
    leader_nodes = nodes[nodes["is_leader"] == True]

    plt.scatter(normal_nodes["x"], normal_nodes["y"], c="lightblue", label="Normal Nodes")
    plt.scatter(ch_nodes["x"], ch_nodes["y"], c="black", s=120, label="Cluster Heads")
    plt.scatter(leader_nodes["x"], leader_nodes["y"], c="green", s=180, label="Leader")
    plt.scatter(SINK_X, SINK_Y, c="red", marker="*", s=200, label="Base Station")

    for i in range(len(chain) - 1):
        x1, y1 = nodes.at[chain[i], "x"], nodes.at[chain[i], "y"]
        x2, y2 = nodes.at[chain[i + 1], "x"], nodes.at[chain[i + 1], "y"]
        plt.plot([x1, x2], [y1, y2], "k--")

    if len(chain) > 0:
        lx, ly = nodes.at[chain[-1], "x"], nodes.at[chain[-1], "y"]
        plt.plot([lx, SINK_X], [ly, SINK_Y], "r-")

    plt.xlim(0, AREA_X)
    plt.ylim(0, AREA_Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_simulation_results(results):
    rounds = np.arange(1, len(results["alive_history"]) + 1)

    plt.figure()
    plt.plot(rounds, results["dead_history"])
    plt.xlabel("Rounds")
    plt.ylabel("Dead Nodes")
    plt.title("Dead Nodes vs Rounds")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(rounds, results["alive_history"])
    plt.xlabel("Rounds")
    plt.ylabel("Alive Nodes")
    plt.title("Alive Nodes vs Rounds")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(rounds, results["avg_energy_history"])
    plt.xlabel("Rounds")
    plt.ylabel("Average Residual Energy")
    plt.title("Average Energy vs Rounds")
    plt.grid(True)
    plt.show()


# =========================================================
# 15. ONE-ROUND TEST
# =========================================================
def run_one_round_demo():
    nodes = initialize_nodes()
    nodes = assign_clusters(nodes)
    nodes["dist_to_sink"] = nodes.apply(distance_to_sink, axis=1)

    plot_network(nodes, title="Initial Network")
    plot_clusters(nodes, title="Clustered Network")

    nodes, ch_indices = select_cluster_heads(nodes)
    plot_cluster_heads(nodes, title="Selected Cluster Heads")

    nodes, leader_idx = select_leader(nodes, ch_indices)
    chain = build_ch_chain(nodes, ch_indices, leader_idx)
    plot_chain(nodes, chain, title="P-LEACH Chain Before Transmission")

    nodes = intra_cluster_transmission(nodes)
    nodes = inter_cluster_transmission(nodes, chain)
    nodes = update_dead_nodes(nodes)

    return nodes, ch_indices, leader_idx, chain


# =========================================================
# 16. FULL MULTI-ROUND SIMULATION
# =========================================================
def run_p_leach_simulation(num_rounds=NUM_ROUNDS, save_features=True):
    nodes = initialize_nodes()

    alive_history = []
    dead_history = []
    avg_energy_history = []

    first_dead_round = None
    half_dead_round = None
    last_dead_round = None

    feature_frames = []

    for r in range(num_rounds):
        # Update sink distance
        nodes["dist_to_sink"] = nodes.apply(distance_to_sink, axis=1)

        # Assign clusters
        nodes = assign_clusters(nodes)

        # CH selection
        nodes, ch_indices = select_cluster_heads(nodes)

        # Leader selection
        nodes, leader_idx = select_leader(nodes, ch_indices)

        # Chain formation
        chain = build_ch_chain(nodes, ch_indices, leader_idx)

        # Save node features for future AI model
        if save_features:
            feature_frames.append(collect_node_features(nodes, r + 1))

        # Communication
        nodes = intra_cluster_transmission(nodes)
        nodes = inter_cluster_transmission(nodes, chain)

        # Death update
        nodes = update_dead_nodes(nodes)

        # Metrics
        alive_count = nodes["alive"].sum()
        dead_count = len(nodes) - alive_count
        avg_energy = nodes["energy"].mean()

        alive_history.append(alive_count)
        dead_history.append(dead_count)
        avg_energy_history.append(avg_energy)

        if first_dead_round is None and dead_count >= 1:
            first_dead_round = r + 1

        if half_dead_round is None and dead_count >= len(nodes) / 2:
            half_dead_round = r + 1

        if alive_count == 0:
            last_dead_round = r + 1
            break

    feature_dataset = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()

    return {
        "nodes": nodes,
        "alive_history": alive_history,
        "dead_history": dead_history,
        "avg_energy_history": avg_energy_history,
        "first_dead_round": first_dead_round,
        "half_dead_round": half_dead_round,
        "last_dead_round": last_dead_round,
        "feature_dataset": feature_dataset
    }


# =========================================================
# 17. RUN EVERYTHING
# =========================================================
# One-round visual demo
demo_nodes, demo_ch_indices, demo_leader_idx, demo_chain = run_one_round_demo()

# Full simulation
results = run_p_leach_simulation()

print("First Node Dies (FND):", results["first_dead_round"])
print("Half Nodes Dead (HND):", results["half_dead_round"])
print("Last Node Dies (LND):", results["last_dead_round"])

plot_simulation_results(results)

# Optional: inspect AI-ready training-style data
print("\nFeature dataset preview:")
print(results["feature_dataset"].head())

# Optional: save features for future ML work
# results["feature_dataset"].to_csv("p_leach_features.csv", index=False)
