import numpy as np
import pandas as pd

def npz_to_csv(npz_path, csv_path):
    data = np.load(npz_path, allow_pickle=True)

    # --- metadata (scalar fields) ---
    meta_keys = [
        "instance_number",
        "N_vertices_full",
        "p_edges",
        "N_vertices_in_LCC",
        "N_processes"
    ]

    meta = {k: data[k].item() if k in data else None for k in meta_keys}

    time_grid = data["time_grid"]

    def safe_get(key):
        return data[key] if key in data else None

    mean_fractions = safe_get("mean_fractions")
    var_fractions = safe_get("var_fractions")

    mean_edge_type = safe_get("mean_edge_type_counts_in_time_in_one_instance")
    var_edge_type = safe_get("var_edge_type_counts")

    mean_tripoint_type = safe_get("mean_tripoint_type_counts_in_time_in_one_instance")
    var_tripoint_type = safe_get("var_tripoint_type_counts")

    mean_1_edge = safe_get("mean_num_of_1_edge_causal_in_time_in_one_instance")
    var_1_edge = safe_get("var_num_of_1_edge_causal_in_time_in_one_instance")

    mean_2_chains = safe_get("mean_num_of_2_chains_in_time_in_one_instance")
    var_2_chains = safe_get("var_num_of_2_chains_in_time_in_one_instance")

    mean_2_instars = safe_get("mean_num_of_2_instars_in_time_in_one_instance")
    var_2_instars = safe_get("var_num_of_2_instars_in_time_in_one_instance")

    mean_2_outstars = safe_get("mean_num_of_2_outstars_in_time_in_one_instance")
    var_2_outstars = safe_get("var_num_of_2_outstars_in_time_in_one_instance")

    rows = []

    for t_idx, t in enumerate(time_grid):
        row = dict(meta)
        row["time"] = t

        # --- main fractions ---
        if mean_fractions is not None:
            row["mean_S"] = mean_fractions[t_idx, 0]
            row["mean_I"] = mean_fractions[t_idx, 1] if mean_fractions.shape[1] > 1 else None

        if var_fractions is not None:
            row["var_S"] = var_fractions[t_idx, 0]
            row["var_I"] = var_fractions[t_idx, 1] if var_fractions.shape[1] > 1 else None

        # --- edge stats ---
        if mean_edge_type is not None:
            for i in range(mean_edge_type.shape[1]):
                row[f"mean_edge_type_{i}"] = mean_edge_type[t_idx, i]
        if var_edge_type is not None:
            for i in range(var_edge_type.shape[1]):
                row[f"var_edge_type_{i}"] = var_edge_type[t_idx, i]

        # --- tripoint stats ---
        if mean_tripoint_type is not None:
            for i in range(mean_tripoint_type.shape[1]):
                row[f"mean_tripoint_type_{i}"] = mean_tripoint_type[t_idx, i]
        if var_tripoint_type is not None:
            for i in range(var_tripoint_type.shape[1]):
                row[f"var_tripoint_type_{i}"] = var_tripoint_type[t_idx, i]

        # --- causal motifs ---
        if mean_1_edge is not None:
            row["mean_1_edge_causal"] = mean_1_edge[t_idx]
        if var_1_edge is not None:
            row["var_1_edge_causal"] = var_1_edge[t_idx]

        if mean_2_chains is not None:
            row["mean_2_chains"] = mean_2_chains[t_idx]
        if var_2_chains is not None:
            row["var_2_chains"] = var_2_chains[t_idx]

        if mean_2_instars is not None:
            row["mean_2_instars"] = mean_2_instars[t_idx]
        if var_2_instars is not None:
            row["var_2_instars"] = var_2_instars[t_idx]

        if mean_2_outstars is not None:
            row["mean_2_outstars"] = mean_2_outstars[t_idx]
        if var_2_outstars is not None:
            row["var_2_outstars"] = var_2_outstars[t_idx]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV → {csv_path}")