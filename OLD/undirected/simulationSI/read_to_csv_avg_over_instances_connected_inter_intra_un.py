import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
simID = "SIsimUNDIRECTED20260421160626" #SIsim20260401112526
Nv = "20"
k = "2"

curves_dir = Path(f"/home/lnf/Desktop/00_sim_SI/{simID}/N{Nv}_k{k}/Curves")

# -----------------------
# LOAD FILTERED AGGREGATE FILE
# -----------------------
# Make sure this is the filtered file (±2% Nconnected)
agg_file = curves_dir / f"curves_average_filtered_inter_intra_{simID}_N{Nv}_k{k}.npz"
data = np.load(agg_file)

# -----------------------
# FLATTEN ARRAYS
# -----------------------
time = data["time_grid"]
mean_fractions = data["mean_fractions"]       # shape (T, n_states)
var_fractions = data["var_fractions"]        # shape (T, n_states)
var_fractions_intra = data["var_fractions_intra"]  # shape (T, n_states)
var_fractions_inter = data["var_fractions_inter"]  # shape (T, n_states)
mean_1edge = data["mean_num_of_1_edge_causal"]
var_1edge = data["var_num_of_1_edge_causal"]
mean_2chains = data["mean_num_of_2_chains"]
var_2chains = data["var_num_of_2_chains"]
mean_2instars = data["mean_num_of_2_instars"]
var_2instars = data["var_num_of_2_instars"]
mean_2outstars = data["mean_num_of_2_outstars"]
var_2outstars = data["var_num_of_2_outstars"]

# Connected component sizes (if stored)
if "mean_N_connected" in data and "var_N_connected" in data:
    mean_N_connected = data["mean_N_connected"]
    var_N_connected = data["var_N_connected"]
else:
    mean_N_connected = np.nan
    var_N_connected = np.nan

n_states = mean_fractions.shape[1]

# Flatten fractions per state
mean_frac_cols = {f"mean_fraction_state{i}": mean_fractions[:, i] for i in range(n_states)}
var_frac_cols = {f"var_fraction_state{i}": var_fractions[:, i] for i in range(n_states)}
var_frac_inter_cols = {f"var_fraction_state{i}": var_fractions_inter[:, i] for i in range(n_states)}
var_frac_intra_cols = {f"var_fraction_state{i}": var_fractions_intra[:, i] for i in range(n_states)}


# Build DataFrame
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
    **var_frac_intra_cols
})

# -----------------------
# SAVE TO CSV
# -----------------------
out_csv = curves_dir / f"curves_average_filtered_inter_intra_{simID}_N{Nv}_k{k}.csv"

# Write metadata in header as comments
metadata_lines = [
    f"# simID={simID}",
    f"# Nv={Nv}",
    f"# k={k}",
    f"# source_file={agg_file.name}",
    f"# mean_N_connected={mean_N_connected}",
    f"# var_N_connected={var_N_connected}"
]

with open(out_csv, "w") as f:
    for line in metadata_lines:
        f.write(line + "\n")
    df.to_csv(f, index=False)

print(f"Saved filtered CSV with flattened arrays to: {out_csv}")