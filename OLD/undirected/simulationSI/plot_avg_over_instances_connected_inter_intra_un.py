import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------
# Simulation identifiers
# -------------------


simID = "SIsimUNDIRECTED20260415095123" #SIsim20260401112526
Nv = "1000"
k = "4"

curves_dir = Path(f"/home/lnf/Desktop/00_sim_SI/{simID}/N{Nv}_k{k}/Curves")

# <-- Use the filtered aggregation file created by the filtering script -->
agg_file = curves_dir / f"curves_average_filtered_inter_intra_{simID}_N{Nv}_k{k}.npz"

# -------------------
# Time interval to cut
# -------------------
t1 = 0.0  # start time
t2 = 20.0  # end time

# -------------------
# Load data
# -------------------
data = np.load(agg_file)

time_grid = data["time_grid"]

# Fractions
mean_fractions = data["mean_fractions"]
var_fractions = data["var_fractions"]  # we will use this for state 0 variance
var_fractions_intra = data["var_fractions_intra"]
var_fractions_inter = data["var_fractions_inter"]
std_fractions = np.sqrt(var_fractions)

# Edge/causal stats
mean_1edge = data["mean_num_of_1_edge_causal"]
std_1edge = np.sqrt(data["var_num_of_1_edge_causal"])

mean_2chains = data["mean_num_of_2_chains"]
std_2chains = np.sqrt(data["var_num_of_2_chains"])

mean_2instars = data["mean_num_of_2_instars"]
std_2instars = np.sqrt(data["var_num_of_2_instars"])

mean_2outstars = data["mean_num_of_2_outstars"]
std_2outstars = np.sqrt(data["var_num_of_2_outstars"])

# -------------------
# Apply time cut
# -------------------
mask = (time_grid >= t1) & (time_grid <= t2)

time_grid_cut = time_grid[mask]
mean_fractions_cut = mean_fractions[mask]
std_fractions_cut = std_fractions[mask]
var_fractions_cut = var_fractions[mask]
var_fractions_intra_cut = var_fractions_intra[mask]
var_fractions_inter_cut = var_fractions_inter[mask]

mean_1edge_cut = mean_1edge[mask]
std_1edge_cut = std_1edge[mask]

mean_2chains_cut = mean_2chains[mask]
std_2chains_cut = std_2chains[mask]

mean_2instars_cut = mean_2instars[mask]
std_2instars_cut = std_2instars[mask]

mean_2outstars_cut = mean_2outstars[mask]
std_2outstars_cut = std_2outstars[mask]

# -------------------
# Plot fractions
# -------------------
plt.figure(figsize=(8, 5))
for state in range(mean_fractions_cut.shape[1]):
    plt.plot(time_grid_cut, mean_fractions_cut[:, state], label=f"State {state}")
    plt.fill_between(
        time_grid_cut,
        mean_fractions_cut[:, state] - std_fractions_cut[:, state],
        mean_fractions_cut[:, state] + std_fractions_cut[:, state],
        alpha=0.2
    )
plt.xlabel("Time")
plt.ylabel("Fraction of vertices")
plt.title(f"Mean fractions ± std (N={Nv}, k={k}) — filtered files")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------
# Plot causal edge stats
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(time_grid_cut, mean_1edge_cut, label="1-edge causal")
plt.fill_between(time_grid_cut, mean_1edge_cut - std_1edge_cut, mean_1edge_cut + std_1edge_cut, alpha=0.2)

plt.plot(time_grid_cut, mean_2chains_cut, label="2-chains")
plt.fill_between(time_grid_cut, mean_2chains_cut - std_2chains_cut, mean_2chains_cut + std_2chains_cut, alpha=0.2)

plt.plot(time_grid_cut, mean_2instars_cut, label="2-instars")
plt.fill_between(time_grid_cut, mean_2instars_cut - std_2instars_cut, mean_2instars_cut + std_2instars_cut, alpha=0.2)

plt.plot(time_grid_cut, mean_2outstars_cut, label="2-outstars")
plt.fill_between(time_grid_cut, mean_2outstars_cut - std_2outstars_cut, mean_2outstars_cut + std_2outstars_cut, alpha=0.2)

plt.xlabel("Time")
plt.ylabel("Number of edges")
plt.title(f"Causal edge statistics ± std (N={Nv}, k={k}) — filtered files")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------
# Plot variance of state 0 separately
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(time_grid_cut, var_fractions_cut[:, 0], color='red', label="Variance of fraction in state 0")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.title(f"Variance of state 0 (N={Nv}, k={k}) — filtered files")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(time_grid_cut, var_fractions_intra_cut[:, 0],
         color='red',
         label="Variance of fraction in state 0, part due to mean of vars (intra)")

plt.plot(time_grid_cut, var_fractions_inter_cut[:, 0],
         color='blue',
         label="Variance of fraction in state 0, part due to var of means (inter)")

plt.plot(time_grid_cut, var_fractions_cut[:, 0],
         color='black',
         linestyle='--',
         label="Total variance of fraction in state 0")

plt.xlabel("Time", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.title(f"Variance of state 0 (N={Nv}, k={k}) — filtered files", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)

plt.tight_layout()
plt.show()

#######################

plt.figure(figsize=(8, 5))

plt.plot(time_grid_cut, var_fractions_inter_cut[:, 0],
         color='blue',
         label="Variance of fraction in state 0, part due to var of means (inter)")


plt.xlabel("Time", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.title(f"Variance of state 0 (N={Nv}, k={k}) — filtered files", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)

plt.tight_layout()
plt.show()