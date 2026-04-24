import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
simID = "SIsimUNDIRECTED20260424101341" #SIsim20260401112526
Nv = "20"
k = "2"

curves_dir = Path(f"/home/lnf/Desktop/00_sim_SI/{simID}/N{Nv}_k{k}/Curves")

# -----------------------
# LOAD FILE
# -----------------------
agg_file = curves_dir / f"curves_average_filtered_inter_intra_Skm_{simID}_N{Nv}_k{k}.npz"
data = np.load(agg_file)

# -----------------------
# LOAD ARRAYS
# -----------------------
time = data["time_grid"]

mean_fractions = data["mean_fractions"]
var_fractions = data["var_fractions"]
var_fractions_intra = data["var_fractions_intra"]
var_fractions_inter = data["var_fractions_inter"]

mean_Skm = data["mean_Skm"]   # (T, K+1, K+1)
var_Skm  = data["var_Skm"]

mean_1edge = data["mean_num_of_1_edge_causal"]
var_1edge = data["var_num_of_1_edge_causal"]

mean_2chains = data["mean_num_of_2_chains"]
var_2chains = data["var_num_of_2_chains"]

mean_2instars = data["mean_num_of_2_instars"]
var_2instars = data["var_num_of_2_instars"]

mean_2outstars = data["mean_num_of_2_outstars"]
var_2outstars = data["var_num_of_2_outstars"]

# -----------------------
# METADATA
# -----------------------
if "mean_N_connected" in data and "var_N_connected" in data:
    mean_N_connected = data["mean_N_connected"]
    var_N_connected = data["var_N_connected"]
else:
    mean_N_connected = np.nan
    var_N_connected = np.nan

# -----------------------
# BASIC DIMENSIONS
# -----------------------
n_states = mean_fractions.shape[1]
T, Kp1, Mp1 = mean_Skm.shape

# -----------------------
# FRACTIONS → COLUMNS
# -----------------------
mean_frac_cols = {f"mean_fraction_state{i}": mean_fractions[:, i] for i in range(n_states)}
var_frac_cols = {f"var_fraction_state{i}": var_fractions[:, i] for i in range(n_states)}
var_frac_inter_cols = {f"var_fraction_state{i}_inter": var_fractions_inter[:, i] for i in range(n_states)}
var_frac_intra_cols = {f"var_fraction_state{i}_intra": var_fractions_intra[:, i] for i in range(n_states)}

# -----------------------
# SPARSE Skm FLATTENING
# -----------------------
mean_Skm_sparse_cols = {}
var_Skm_sparse_cols = {}

print("Flattening Skm (sparse)...")

for k_val in range(Kp1):
    for m_val in range(Mp1):
        mean_series = mean_Skm[:, k_val, m_val]

        # keep only nonzero trajectories
        if np.any(mean_series != 0):
            col_mean = f"mean_Skm_k{k_val}_m{m_val}"
            col_var  = f"var_Skm_k{k_val}_m{m_val}"

            mean_Skm_sparse_cols[col_mean] = mean_series
            var_Skm_sparse_cols[col_var]   = var_Skm[:, k_val, m_val]

print(f"Kept {len(mean_Skm_sparse_cols)} Skm columns (sparse)")

# -----------------------
# BUILD DATAFRAME
# -----------------------
df = pd.DataFrame({
    "time": time,

    "mean_num_of_1_edge_causal": mean_1edge,
    "var_num_of_1_edge_causal": var_1edge,

    "mean_num_of_2_chains": mean_2chains,
    "var_num_of_2_chains": var_2chains,

    "mean_num_of_2_instars": mean_2instars,
    "var_num_of_2_instars": var_2instars,

    "mean_num_of_2_outstars": mean_2outstars,
    "var_num_of_2_outstars": var_2outstars,

    **mean_frac_cols,
    **var_frac_cols,
    **var_frac_inter_cols,
    **var_frac_intra_cols,

    **mean_Skm_sparse_cols,
    **var_Skm_sparse_cols
})

# -----------------------
# SAVE CSV
# -----------------------
out_csv = curves_dir / f"curves_average_filtered_inter_intra_Skm_{simID}_N{Nv}_k{k}.csv"

metadata_lines = [
    f"# simID={simID}",
    f"# Nv={Nv}",
    f"# k={k}",
    f"# source_file={agg_file.name}",
    f"# mean_N_connected={mean_N_connected}",
    f"# var_N_connected={var_N_connected}",
    f"# Skm_sparse_columns={len(mean_Skm_sparse_cols)}"
]

with open(out_csv, "w") as f:
    for line in metadata_lines:
        f.write(line + "\n")
    df.to_csv(f, index=False)

print(f"Saved CSV to: {out_csv}")