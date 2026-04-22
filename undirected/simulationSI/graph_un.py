# current

import numpy as np
import graph_tool.all as gt
from collections import Counter

def generate_undirected_ER_graph(N_vertices, p_edges):
    # Step 1: generate all possible edges (upper triangular, i < j)
    possible_v1, possible_v2 = np.triu_indices(N_vertices, k=1)
    
    # Step 2: randomly decide which edges exist
    mask = np.random.rand(len(possible_v1)) < p_edges
    v1 = possible_v1[mask]
    v2 = possible_v2[mask]
    
    # Step 3: sort edges lexicographically by v1, then v2>v1
    order = np.lexsort((v2, v1)) # they should technically already be sorted like this, this is defensive
    v1_sorted = v1[order]
    v2_sorted_by_v1 = v2[order]
    
    return v1_sorted, v2_sorted_by_v1

#################

def take_largest_connected_component(N_vertices_full, v1, v2):
    """
    Extract largest connected component from an undirected graph.
    Returns relabeled (0..N_connected-1) graph.
    """

    # Build undirected graph
    g = gt.Graph(directed=False)
    g.add_vertex(N_vertices_full)

    edges = np.vstack((v1, v2)).T
    g.add_edge_list(edges)

    # Connected components (no need for directed=False anymore)
    comp, hist = gt.label_components(g)
    largest_comp = np.argmax(hist)

    # Boolean mask of vertices in LCC
    mask_vertices = (comp.a == largest_comp)
    N_connected = int(mask_vertices.sum())

    if N_connected == 0:
        return 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Relabel nodes
    old_to_new = -np.ones(N_vertices_full, dtype=np.int32)
    old_to_new[mask_vertices] = np.arange(N_connected, dtype=np.int32)

    # Keep only edges inside LCC
    mask_edges = mask_vertices[v1] & mask_vertices[v2]

    v1_new = old_to_new[v1[mask_edges]]
    v2_new = old_to_new[v2[mask_edges]]

    return N_connected, v1_new, v2_new

def v1_pointer(v1_sorted, N_vertices):
    counts = np.bincount(v1_sorted, minlength=N_vertices)

    ptr = np.zeros(N_vertices + 1, dtype=int)
    ptr[1:] = np.cumsum(counts)

    return ptr # neighbors_of_i = v2_sorted_by_v1[ptr[i]:ptr[i+1]]

def sort_v1_by_v2(v1_sorted, v2_sorted_by_v1):
    E = len(v1_sorted)
    edge_ids = np.arange(E)

    idx = np.lexsort((v1_sorted, v2_sorted_by_v1))

    v1_sorted_by_v2 = v1_sorted[idx]
    v2_sorted = v2_sorted_by_v1[idx]
    edge_ids_sorted_by_v2 = edge_ids[idx]

    return v1_sorted_by_v2, v2_sorted, edge_ids_sorted_by_v2

def v2_pointer(v2_sorted, N_vertices):
    counts = np.bincount(v2_sorted, minlength=N_vertices)

    ptr = np.zeros(N_vertices + 1, dtype=int)
    ptr[1:] = np.cumsum(counts)

    return ptr # neighbors_of_i = v1_sorted_by_v2[ptr[i]:ptr[i+1]]


