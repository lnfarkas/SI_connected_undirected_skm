# current

import numpy as np

# change update rules here
def change_target_state_by_contact_process(vertex_states,src,tgt):
    vertex_states[tgt]=vertex_states[src]

def change_vertex_state_by_vertex_event(vertex_states,vertex,state):
    vertex_states[vertex] = state
# update rules end

def prep_rates(rates):
    allowed = rates > 0
    inv_rates = np.full_like(rates, np.inf, dtype=np.float64)
    inv_rates[allowed] = 1.0 / rates[allowed]
    return allowed, inv_rates

def initial_vertex_states_SI(
    N_vertices,
    fraction_per_state_sans_state_0_initial
):
    fractions = np.asarray(fraction_per_state_sans_state_0_initial)
    n_states = fractions.size + 1  # include state 0

    vertex_states = np.zeros(N_vertices, dtype=int)

    counts_nonzero = np.floor(fractions * N_vertices).astype(int)
    states_nonzero = np.arange(1, n_states)

    v_states = np.repeat(states_nonzero, counts_nonzero)
    np.random.shuffle(v_states)

    perm = np.random.permutation(N_vertices)

    chosen_vertices = perm[:v_states.size]
    vertex_states[chosen_vertices] = v_states

    return vertex_states # BE AWARE THIS IS NOT HOW IT WAS DONE BEFORE

def find_edge_types(v1_sorted, v2_sorted_by_v1, vertex_states, n_states):

    s1 = vertex_states[v1_sorted]
    s2 = vertex_states[v2_sorted_by_v1]

    s_min = np.minimum(s1, s2)
    s_max = np.maximum(s1, s2)

    edge_type = s_min * n_states - (s_min * (s_min - 1)) // 2 + (s_max - s_min)
    return edge_type


def init_edge_events(edge_types, allowed_edges, inv_edge_rates):
    E = len(edge_types)
    edge_events = np.zeros((E, 2), dtype=np.float64)
    edge_events[:, 0] = np.arange(E)           
    edge_events[:, 1] = np.inf                 

    allowed_mask = allowed_edges[edge_types]
    edge_events[allowed_mask, 1] = np.random.exponential(
        scale=inv_edge_rates[edge_types[allowed_mask]],
        size=allowed_mask.sum()
    )

    return edge_events

def init_vertex_events(vertex_states, allowed_vertices, inv_vertex_rates, N_edges):
    N_vertices = len(vertex_states)
    vertex_events = np.zeros((N_vertices, 2), dtype=np.float64)
    vertex_events[:, 0] = N_edges + np.arange(N_vertices) # we later concatenate with edge events 
    vertex_events[:, 1] = np.inf

    allowed_mask = allowed_vertices[vertex_states]
    vertex_events[allowed_mask, 1] = np.random.exponential(
        scale=inv_vertex_rates[vertex_states[allowed_mask]],
        size=allowed_mask.sum()
    )

    return vertex_events



def update_vertex_event_time(
    vertex,
    vertex_states,
    events,
    allowed_vertices,
    inv_vertex_rates,
    N_edges,
    current_time
    ):
    idx = N_edges + vertex
    events[idx, 1] = np.inf

    state = vertex_states[vertex]
    if allowed_vertices[state]:
        events[idx, 1] = current_time + np.random.exponential(
            scale=inv_vertex_rates[state]
        )

### here

def update_edges_after_vertex_change( # this does not depend on the update rules
    vertex, # the one thats experiencing the change
    vertex_states,
    targets_sorted_by_source,
    sources_sorted_by_target,
    edge_ids_sorted_by_target,
    ptr_src,
    ptr_tgt,
    n_states,
    edge_types,
    events,
    allowed,
    inv_rates,
    current_time,
    causal,
    causal_in,
    causal_out
    ):
    vertex_state = vertex_states[vertex]
    s0, s1 = ptr_src[vertex], ptr_src[vertex + 1]
    if s1 > s0:
        tgt_states = vertex_states[targets_sorted_by_source[s0:s1]]

        new_types = vertex_state * n_states + tgt_states
        edge_types[s0:s1] = new_types

        events[s0:s1, 1] = np.inf
        mask = allowed[new_types]
        if mask.any():
            events[s0:s1, 1][mask] = current_time + np.random.exponential(
                scale=inv_rates[new_types[mask]],
                size=mask.sum()
            )
        if vertex_state == 0:
            causal[s0:s1] = 0

    
    t0, t1 = ptr_tgt[vertex], ptr_tgt[vertex + 1]
    if t1 > t0:
        src_states = vertex_states[sources_sorted_by_target[t0:t1]]

        new_types = src_states * n_states + vertex_state

        e_ids = edge_ids_sorted_by_target[t0:t1] 
        # mapping from target-sorted ptr_tgt 
        # to "standard" source-sorted indices

        edge_types[e_ids] = new_types
        events[e_ids, 1] = np.inf

        mask = allowed[new_types]
        if mask.any():
            events[e_ids[mask], 1] = current_time + np.random.exponential(
                scale=inv_rates[new_types[mask]],
                size=mask.sum()
            )
        new_causal_mask = (vertex_state != 0) & (src_states != 0)
        causal[e_ids] =  new_causal_mask
        # check that the new vertex is infected (not recovered) and all sources that have been already infected then form a causal edge to it
        num_new_1_edge_causal = np.sum(new_causal_mask)  # edges ending in 'vertex' that just became causal # WRONG FOR SIS/SIR, WORKS ONLY FOR SI
        causal_in[vertex] = num_new_1_edge_causal   # WRONG FOR SIS/SIR, WORKS ONLY FOR SI
        sources = sources_sorted_by_target[t0:t1][new_causal_mask]
        causal_out[sources] += 1
        num_new_2_chains = 0
        num_new_2_instars = 0
        num_new_2_outstars = 0
        if sources.size > 0: # WRONG FOR SIS/SIR, WORKS ONLY FOR SI
            num_new_2_chains = np.sum(causal_in[sources])
            num_new_2_instars = num_new_1_edge_causal*(num_new_1_edge_causal-1)/2
            num_new_2_outstars = np.sum(causal_out[sources]-1)
        else:
            num_new_2_chains = 0
            num_new_2_instars = 0
            num_new_2_outstars = 0
    else:
        num_new_1_edge_causal = 0 # WRONG FOR SIS/SIR, WORKS ONLY FOR SI
        num_new_2_chains = 0
        num_new_2_instars = 0
        num_new_2_outstars = 0

    return num_new_1_edge_causal, num_new_2_chains, num_new_2_instars, num_new_2_outstars



def step(
    causal,
    causal_in,
    causal_out,
    N_edges,
    vertex_states,
    events,
    edge_types,
    allowed_edges,
    inv_edge_rates,
    allowed_vertices,
    inv_vertex_rates,
    sources_sorted,
    targets_sorted_by_source,
    sources_sorted_by_target,
    edge_ids_sorted_by_target,
    ptr_src,
    ptr_tgt,
    n_states,
    current_time,
    current_counts
):

    allowed_edge_mask = allowed_edges[edge_types]
    allowed_vertex_mask = allowed_vertices[vertex_states]

    allowed_mask = np.concatenate((allowed_edge_mask, allowed_vertex_mask))

    if not allowed_mask.any():
        return 0, current_time, current_counts, 0, 0, 0, 0
        # returns 0 if no events are possible, SECOND 0 IS FOR NEW CAUSAL EDGES - THIS IS WRONG FOR SIS/SIR, WORKS ONLY FOR SI

    next_idx = np.argmin(
        np.where(allowed_mask, events[:, 1], np.inf)
    )

    current_time = events[next_idx, 1]

    if next_idx < N_edges: # vertex events are numbered as N_edges + vertex index
        src = sources_sorted[next_idx]
        vertex = targets_sorted_by_source[next_idx] # identify affected vertex
        change_target_state_by_contact_process(vertex_states,src,vertex)
    else:
        vertex = next_idx - N_edges # vertex events are numbered as N_edges + vertex index
        change_vertex_state_by_vertex_event(vertex_states, vertex, 0) # 0 -> recovery



    update_vertex_event_time(vertex, vertex_states, events, allowed_vertices, inv_vertex_rates, N_edges, current_time)
    num_new_1_edge_causal, num_new_2_chains, num_new_2_instars, num_new_2_outstars = update_edges_after_vertex_change(
                                                                        vertex,
                                                                        vertex_states,
                                                                        targets_sorted_by_source,
                                                                        sources_sorted_by_target,
                                                                        edge_ids_sorted_by_target,
                                                                        ptr_src,
                                                                        ptr_tgt,
                                                                        n_states,
                                                                        edge_types,
                                                                        events,
                                                                        allowed_edges,
                                                                        inv_edge_rates,
                                                                        current_time,
                                                                        causal,
                                                                        causal_in,
                                                                        causal_out
                                                                        )

    current_counts = np.bincount(vertex_states, minlength=n_states)

    return 1, current_time, current_counts, num_new_1_edge_causal, num_new_2_chains, num_new_2_instars, num_new_2_outstars

