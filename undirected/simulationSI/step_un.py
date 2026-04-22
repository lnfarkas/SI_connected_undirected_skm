
# current

from curses.ascii import SI

import numpy as np

# change update rules here
def change_state_by_contact_process(vertex_states,v_affecting,v_affected):
    vertex_states[v_affected]=vertex_states[v_affecting]

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
    vertices_eligible_for_initial_infection,
    fraction_per_state_sans_state_0_initial
):
    fractions = np.asarray(fraction_per_state_sans_state_0_initial)
    n_states = fractions.size + 1  # include state 0

    vertex_states = np.zeros(N_vertices, dtype=int)  # everything starts as 0

    counts_nonzero = np.round(fractions * N_vertices).astype(int)
    states_nonzero = np.arange(1, n_states)
    post_initial_infection_states = np.repeat(states_nonzero, counts_nonzero)
    np.random.shuffle(post_initial_infection_states)

    chosen_vertices = np.random.choice(vertices_eligible_for_initial_infection, size=post_initial_infection_states.size, replace=False)
    vertex_states[chosen_vertices] = post_initial_infection_states
    return vertex_states

def find_edge_types(v1_sorted, v2_sorted_by_v1, vertex_states, n_states):
    v1_states = vertex_states[v1_sorted]
    v2_states = vertex_states[v2_sorted_by_v1]

    a = np.minimum(v1_states, v2_states)
    b = np.maximum(v1_states, v2_states)

    return b * (b + 1) // 2 + a


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



def undirected_edge_type(a, b):
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return hi * (hi + 1) // 2 + lo


def update_edges_after_vertex_change_undirected(
    vertex,
    vertex_states,
    v1_sorted,
    v2_sorted_by_v1,
    v1_sorted_by_v2,
    edge_ids_sorted_by_v2,
    deg, m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,
    skm,
    ptr_v1,
    ptr_v2,
    n_states,
    n_edge_types,
    edge_types,
    events,
    allowed,
    inv_rates,
    current_time,
    causal,
    causal_in,
    causal_out
):
    v_state = vertex_states[vertex]

    # =========================================================
    # 1. edges where vertex is v1
    # =========================================================
    s0, s1 = ptr_v1[vertex], ptr_v1[vertex + 1]

    if s1 > s0:
        nbr = v2_sorted_by_v1[s0:s1]
        nbr_states = vertex_states[nbr]

        new_types = undirected_edge_type(v_state, nbr_states)

        edge_types[s0:s1] = new_types
        events[s0:s1, 1] = np.inf

        mask = allowed[new_types]
        if mask.any():
            events[s0:s1, 1][mask] = current_time + np.random.exponential(
                scale=inv_rates[new_types[mask]],
                size=mask.sum()
            )

        if v_state == 0:
            causal[s0:s1] = 0

    # =========================================================
    # 2. edges where vertex is v2
    # =========================================================
    t0, t1 = ptr_v2[vertex], ptr_v2[vertex + 1]

    num_new_1_edge_causal = 0
    num_new_2_chains = 0
    num_new_2_instars = 0
    num_new_2_outstars = 0

    if t1 > t0:
        src = v1_sorted_by_v2[t0:t1]
        e_ids = edge_ids_sorted_by_v2[t0:t1]

        src_states = vertex_states[src]

        new_types = undirected_edge_type(src_states, v_state)

        edge_types[e_ids] = new_types
        events[e_ids, 1] = np.inf

        mask = allowed[new_types]
        if mask.any():
            events[e_ids[mask], 1] = current_time + np.random.exponential(
                scale=inv_rates[new_types[mask]],
                size=mask.sum()
            )

        # =====================================================
        # causal logic DOESNT MAKE SENSE ATM JUST IGNORE
        # =====================================================
        new_causal_mask = (v_state != 0) & (src_states != 0)

        causal[e_ids] = new_causal_mask

        num_new_1_edge_causal = int(np.sum(new_causal_mask))

        causal_in[vertex] += num_new_1_edge_causal
        causal_out[src[new_causal_mask]] += 1

        if src.size > 0:
            num_new_2_chains = np.sum(causal_in[src[new_causal_mask]]) if np.any(new_causal_mask) else 0
            num_new_2_instars = num_new_1_edge_causal * (num_new_1_edge_causal - 1) / 2
            num_new_2_outstars = np.sum(causal_out[src[new_causal_mask]] - 1) if np.any(new_causal_mask) else 0

    # =========================================================
    # update m (infected neighbor counts) after vertex change
    # But also update S and I and m_i, m_s, degmm_i, degmm_s
    # =========================================================
    if vertex_states[vertex] == 1:
        s0, s1 = ptr_v1[vertex], ptr_v1[vertex + 1]
        if s1 > s0:
            m[v2_sorted_by_v1[s0:s1]] += 1

        t0, t1 = ptr_v2[vertex], ptr_v2[vertex + 1]
        if t1 > t0:
            m[v1_sorted_by_v2[t0:t1]] += 1    

    # =========================================================
    # update skm
    # =========================================================

    return (
        num_new_1_edge_causal,
        num_new_2_chains,
        num_new_2_instars,
        num_new_2_outstars, #THEY ARE WRONG; IGNORE
        m, S, I, m_s, m_i, degmm_s, degmm_s,
        skm
    )


#########################################################################################

def step(
    causal,
    causal_in,
    causal_out,
    N_edges,
    vertex_states,
    skm,
    events,
    edge_types,
    allowed_edges,
    inv_edge_rates,
    allowed_vertices,
    inv_vertex_rates,
    v1_sorted,
    v2_sorted_by_v1,
    v1_sorted_by_v2,
    edge_ids_sorted_by_v2,
    deg, m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,
    ptr_v1,
    ptr_v2,
    n_states,
    n_edge_types,
    current_time,
    current_counts
):

    allowed_edge_mask = allowed_edges[edge_types]
    allowed_vertex_mask = allowed_vertices[vertex_states]

    allowed_mask = np.concatenate((allowed_edge_mask, allowed_vertex_mask))

    if not allowed_mask.any():
        return 0, current_time, current_counts, np.zeros(3, dtype=int),np.zeros(6, dtype=int), 0, 0, 0, 0, m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,skm

    next_idx = np.argmin(np.where(allowed_mask, events[:, 1], np.inf))
    if next_idx >= N_edges:
        raise RuntimeError("Vertex event triggered in SI model — should not happen")
    current_time = events[next_idx, 1]

    # =========================================================
    # EVENT TYPE SELECTION
    # =========================================================

    if next_idx < N_edges:  # edge event
        edge_id = next_idx

        v1 = v1_sorted[edge_id]
        v2 = v2_sorted_by_v1[edge_id]

        s1 = vertex_states[v1]
        s2 = vertex_states[v2]

        # enforce valid contact event: exactly one active site
        if s1 == s2:
            raise ValueError(f"Invalid contact event: s1={s1}, s2={s2}")

        # determine direction: active (1) -> src, inactive (0) -> tgt
        if s1 == 1:
            src, tgt = v1, v2
        else:
            src, tgt = v2, v1

        change_state_by_contact_process(vertex_states, src, tgt)

        # updated vertex is the one that changed state (the target)
        updated_vertex = tgt

    else:
        # vertex event
        vertex = next_idx - N_edges
        change_vertex_state_by_vertex_event(vertex_states, vertex, 0)
        updated_vertex = vertex

    # =========================================================
    # vertex event refresh
    # =========================================================
    update_vertex_event_time(
        updated_vertex,
        vertex_states,
        events,
        allowed_vertices,
        inv_vertex_rates,
        N_edges,
        current_time
    )

    # =========================================================
    # edge updates (UNDIRECTED version)
    # =========================================================
    (
        num_new_1_edge_causal,
        num_new_2_chains,
        num_new_2_instars,
        num_new_2_outstars,
        m, S, I, m_s, m_i, degmm_s, degmm_s,
        skm
    ) = update_edges_after_vertex_change_undirected(
        updated_vertex,
        vertex_states,
        v1_sorted,
        v2_sorted_by_v1,
        v1_sorted_by_v2,
        edge_ids_sorted_by_v2,
        deg, m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,
        skm,
        ptr_v1,
        ptr_v2,
        n_states,
        n_edge_types,
        edge_types,
        events,
        allowed_edges,
        inv_edge_rates,
        current_time,
        causal,
        causal_in,
        causal_out,
    )

    current_counts = np.bincount(vertex_states, minlength=n_states)

    edge_type_current_counts = np.bincount(edge_types, minlength=n_edge_types) # (SS,SI,II)

    tripoint_type_current_counts = np.zeros(6, dtype=int) # (SSS,SSI = ISS,ISI,SIS,SII = IIS ,III) -  ignore for now 
    '''
    degmm = deg - m
    
    S = (vertex_states == 0)
    I = (vertex_states == 1)

    deg_s = deg[S]
    m_s = m[S]
    degmm_s = degmm[S]

    deg_i = deg[I]
    m_i = m[I]
    degmm_i = degmm[I]
    '''
    tripoint_type_current_counts[0] = np.sum(degmm_s * (degmm_s - 1) // 2)
    tripoint_type_current_counts[1] = np.sum(m_s*degmm_s)
    tripoint_type_current_counts[2] = np.sum(m_s*(m_s-1) // 2)
    tripoint_type_current_counts[3] = np.sum(degmm_i * (degmm_i - 1) // 2)
    tripoint_type_current_counts[4] = np.sum(m_i*degmm_i)
    tripoint_type_current_counts[5] = np.sum(m_i*(m_i-1) // 2)

    return (
        1,
        current_time,
        current_counts,
        num_new_1_edge_causal,
        num_new_2_chains,
        num_new_2_instars,
        num_new_2_outstars,   # IGNORE THE CAUSAL STUFF FOR NOW; IT IS WRONG!
        edge_type_current_counts,
        tripoint_type_current_counts,
        m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,skm
        )