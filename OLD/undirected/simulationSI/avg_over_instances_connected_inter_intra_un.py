import numpy as np
from pathlib import Path

# ------------------ User Parameters ------------------
tol = float(input("Enter tolerance for the N in LCC(e.g. 0.02 for ±2%): "))

simID = "SIsimUNDIRECTED20260421160626" #SIsim20260401112526
Nv = "20"
k = "2"

curves_dir = Path(f"/home/lnf/Desktop/00_sim_SI/{simID}/N{Nv}_k{k}/Curves")

# ------------------ Gather Files ------------------
files = list(curves_dir.glob("curves_instanceNo*.npz"))
if not files:
    raise RuntimeError("No curve files found")

# Extract Nconnected from filenames
nconnected_list = []
for f in files:
    stem = f.stem
    # Expect pattern like: graph_ER_directed_instanceNo0005_N1000_Nconnected982_k2.0_SIsim20260401112526
    try:
        nconnected = int(stem.split("Nconnected")[1].split("_")[0])
        nconnected_list.append(nconnected)
    except IndexError:
        raise ValueError(f"Filename {f} does not match expected pattern")

nconnected_arr = np.array(nconnected_list)
mean_nconnected = nconnected_arr.mean()

lower = mean_nconnected * (1 - tol)
upper = mean_nconnected * (1 + tol)

# Filter files within ±2% of mean
accepted_files = [f for f, n in zip(files, nconnected_arr) if lower <= n <= upper]
rejected_count = len(files) - len(accepted_files)

print(f"Mean Nconnected = {mean_nconnected:.2f}")
print(f"Accepted files: {len(accepted_files)}")
print(f"Rejected files: {rejected_count}")

if not accepted_files:
    raise RuntimeError("No files accepted after filtering")

# ------------------ Group by instance ------------------
instance_dict = {}
for f in accepted_files:
    data = np.load(f)
    instance = int(data["instance_number"])
    n_proc = int(data["N_processes"])
    instance_dict.setdefault(instance, []).append((n_proc, f))

# Keep only max N_processes per instance
selected_files = [sorted(items, key=lambda x: x[0])[-1][1] for items in instance_dict.values()]
print(f"Using {len(selected_files)} instances after filtering")

# ------------------ Aggregate Data ------------------
mean_fractions_list, var_fractions_list = [], []
mean_1edge_list, var_1edge_list = [], []
mean_2chains_list, var_2chains_list = [], []
mean_2instars_list, var_2instars_list = [], []
mean_2outstars_list, var_2outstars_list = [], []
N_connected_list = []

for f in selected_files:
    d = np.load(f)
    mean_fractions_list.append(d["mean_fractions"])
    var_fractions_list.append(d["var_fractions"])
    mean_1edge_list.append(d["mean_num_of_1_edge_causal_in_time_in_one_instance"])
    var_1edge_list.append(d["var_num_of_1_edge_causal_in_time_in_one_instance"])
    mean_2chains_list.append(d["mean_num_of_2_chains_in_time_in_one_instance"])
    var_2chains_list.append(d["var_num_of_2_chains_in_time_in_one_instance"])
    mean_2instars_list.append(d["mean_num_of_2_instars_in_time_in_one_instance"])
    var_2instars_list.append(d["var_num_of_2_instars_in_time_in_one_instance"])
    mean_2outstars_list.append(d["mean_num_of_2_outstars_in_time_in_one_instance"])
    var_2outstars_list.append(d["var_num_of_2_outstars_in_time_in_one_instance"])
    N_connected_list.append(d["N_vertices_in_LCC"])

# Convert to arrays
mean_fractions_arr = np.array(mean_fractions_list)
var_fractions_arr = np.array(var_fractions_list)
mean_1edge_arr = np.array(mean_1edge_list)
var_1edge_arr = np.array(var_1edge_list)
mean_2chains_arr = np.array(mean_2chains_list)
var_2chains_arr = np.array(var_2chains_list)
mean_2instars_arr = np.array(mean_2instars_list)
var_2instars_arr = np.array(var_2instars_list)
mean_2outstars_arr = np.array(mean_2outstars_list)
var_2outstars_arr = np.array(var_2outstars_list)
N_connected_arr = np.array(N_connected_list)

# Compute means
mean_fractions = mean_fractions_arr.mean(axis=0)
mean_1edge = mean_1edge_arr.mean(axis=0)
mean_2chains = mean_2chains_arr.mean(axis=0)
mean_2instars = mean_2instars_arr.mean(axis=0)
mean_2outstars = mean_2outstars_arr.mean(axis=0)
mean_N_connected = N_connected_arr.mean()

# Compute variances
var_fractions = var_fractions_arr.mean(axis=0) + mean_fractions_arr.var(axis=0, ddof=1)
var_fractions_intra = var_fractions_arr.mean(axis=0)
var_fractions_inter =  mean_fractions_arr.var(axis=0, ddof=1)
var_1edge = var_1edge_arr.mean(axis=0) + mean_1edge_arr.var(axis=0, ddof=1)
var_2chains = var_2chains_arr.mean(axis=0) + mean_2chains_arr.var(axis=0, ddof=1)
var_2instars = var_2instars_arr.mean(axis=0) + mean_2instars_arr.var(axis=0, ddof=1)
var_2outstars = var_2outstars_arr.mean(axis=0) + mean_2outstars_arr.var(axis=0, ddof=1)
var_N_connected = N_connected_arr.var(ddof=1)

# ------------------ Save Aggregated Results ------------------
out_file = curves_dir / f"curves_average_filtered_inter_intra_{simID}_N{Nv}_k{k}.npz"
np.savez_compressed(
    out_file,
    mean_fractions=mean_fractions,
    var_fractions=var_fractions,
    var_fractions_intra=var_fractions_intra,
    var_fractions_inter=var_fractions_inter,
    mean_num_of_1_edge_causal=mean_1edge,
    var_num_of_1_edge_causal=var_1edge,
    mean_num_of_2_chains=mean_2chains,
    var_num_of_2_chains=var_2chains,
    mean_num_of_2_instars=mean_2instars,
    var_num_of_2_instars=var_2instars,
    mean_num_of_2_outstars=mean_2outstars,
    var_num_of_2_outstars=var_2outstars,
    mean_N_connected=mean_N_connected,
    var_N_connected=var_N_connected,
    time_grid=d["time_grid"]
)
print(f"Saved aggregated results to: {out_file}")