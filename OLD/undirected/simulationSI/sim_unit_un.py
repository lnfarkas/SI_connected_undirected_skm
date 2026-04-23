import numpy as np
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import csv

from graph_un import *
from step_un import *
#from simualtionSI import time_grid
from time_grid import*
from check_disk_space import *


# Worker initializer to lower priority and seed RNG so workers don't overburden the PC
def _worker_initializer(nice_level, allowed_cores, seed_offset=0):
    # lower priority
    try:
        os.nice(nice_level)
    except Exception:
        pass

    # hard pin worker to allowed cores
    try:
        os.sched_setaffinity(0, allowed_cores)
    except AttributeError:
        # non-Linux systems
        pass

    # seed RNG
    seed = (int(time.time() * 1e6) + os.getpid() + int(seed_offset)) % (2**32 - 1)
    np.random.seed(seed)


# Run a single independent realization (safe to execute in a worker process)
def run_one_realization(args):
    (N_vertices_full,N_vertices_in_LCC, v1_sorted,
    v2_sorted_by_v1,
    v1_sorted_by_v2,
    edge_ids_sorted_by_v2,
    deg, 
    ptr_v1,
    ptr_v2, N_edges, n_states,n_edge_types,
     fractions_initial, allowed_edges, inv_edge_rates, allowed_vertices, inv_vertex_rates,
     time_grid_t, N_time_bins, T_max) = args
    



    # INITIALIZE STATES
    vertex_states = initial_vertex_states_SI(N_vertices_in_LCC,np.arange(N_vertices_in_LCC), fractions_initial)
    edge_types = find_edge_types(v1_sorted, v2_sorted_by_v1, vertex_states, n_states)
    edge_events = init_edge_events(edge_types, allowed_edges, inv_edge_rates)
    vertex_events = init_vertex_events(vertex_states, allowed_vertices, inv_vertex_rates, N_edges)

    events = np.vstack((edge_events, vertex_events))

    # SI SPECIFIC
    n_infected_init = sum(vertex_states)

    causal = np.zeros(N_edges, dtype=bool)
    causal_in  = np.zeros(N_vertices_in_LCC, dtype=int)
    causal_out = np.zeros(N_vertices_in_LCC, dtype=int)

    times = []
    counts_in_time = []
    edge_type_counts_in_time = []
    tripoint_type_counts_in_time = []

    num_new_1_edge_causal = 0
    num_new_2_chains = 0
    num_new_2_instars = 0
    num_new_2_outstars = 0
    total_1_edge_causal = 0
    total_2_chains = 0
    total_2_instars = 0
    total_2_outstars = 0
    num_of_1_edge_causal_in_time = []
    num_of_2_chains_in_time = []
    num_of_2_instars_in_time = []
    num_of_2_outstars_in_time = []

    current_time = 0.0
    current_counts = np.bincount(vertex_states, minlength=n_states)
    edge_type_current_counts = np.bincount(edge_types, minlength=n_edge_types) # (SS,SI,II)


    m = np.zeros(N_vertices_in_LCC, dtype=np.int64)

    m[v1_sorted] += vertex_states[v2_sorted_by_v1]
    m[v2_sorted_by_v1] += vertex_states[v1_sorted]

    S = (vertex_states == 0)
    I = (vertex_states == 1)

    degmm = deg - m

    deg_s = deg[S]
    m_s = m[S]
    degmm_s = degmm[S]

    deg_i = deg[I]
    m_i = m[I]
    degmm_i = degmm[I]

    # =====================================================
    # SKM INITIALIZATION (dense, vectorized)
    # =====================================================

    k_max = deg.max()

    # Dense matrix: rows = k, cols = m
    skm = np.zeros((k_max + 1, k_max + 1), dtype=np.int64)

    # Accumulate counts
    np.add.at(skm, (deg_s, m_s), 1)

    '''
    skm = np.empty(k_max + 1, dtype=object)
    for k in range(k_max + 1):
        skm[k] = np.zeros(k + 1, dtype=np.int64)

    for i in S:
        k = deg[i]
        m_i = m[i]
        skm[k][m_i] += 1
    '''
    # for some k, s will be empty, but i do this for easier comparison between realizations and with numerics
    # acess as s[k][m]

    #tripoint_type_current_counts = np.zeros(6, dtype=int) #np.bincount(tripoint_types, minlength=n_tripoint_types) # (SSS,SSI = ISS,ISI,SIS,SII = IIS ,III) -  ignore for now
    tripoint_type_current_counts = np.zeros(6, dtype=int)

    tripoint_type_current_counts[0] = np.sum(degmm_s * (degmm_s - 1) // 2)
    tripoint_type_current_counts[1] = np.sum(m_s*degmm_s)
    tripoint_type_current_counts[2] = np.sum(m_s*(m_s-1) // 2)
    tripoint_type_current_counts[3] = np.sum(degmm_i * (degmm_i - 1) // 2)
    tripoint_type_current_counts[4] = np.sum(m_i*degmm_i)
    tripoint_type_current_counts[5] = np.sum(m_i*(m_i-1) // 2)

    check = 1

    while current_time < T_max and check:
        times.append(current_time)
        counts_in_time.append(current_counts)
        edge_type_counts_in_time.append(edge_type_current_counts)
        tripoint_type_counts_in_time.append(tripoint_type_current_counts)

        total_1_edge_causal += num_new_1_edge_causal
        total_2_chains += num_new_2_chains
        total_2_instars += num_new_2_instars
        total_2_outstars += num_new_2_outstars
        num_of_1_edge_causal_in_time.append(total_1_edge_causal)
        num_of_2_chains_in_time.append(total_2_chains)
        num_of_2_instars_in_time.append(total_2_instars)
        num_of_2_outstars_in_time.append(total_2_outstars)



        (check,
         current_time,
         current_counts,
         num_new_1_edge_causal,
         num_new_2_chains,
         num_new_2_instars,
         num_new_2_outstars,
         edge_type_current_counts,
         tripoint_type_current_counts,
         m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,skm) = step(
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
                                    )

    projected_counts = project_to_time_grid(times, counts_in_time, time_grid_t)
    projected_edge_type_counts = project_to_time_grid(times, edge_type_counts_in_time, time_grid_t)
    projected_tripoint_type_counts = project_to_time_grid(times, tripoint_type_counts_in_time, time_grid_t)
    projected_num_of_1_edge_causal = project_to_time_grid(times, num_of_1_edge_causal_in_time, time_grid_t)
    projected_num_of_2_chains = project_to_time_grid(times, num_of_2_chains_in_time, time_grid_t)
    projected_num_of_2_instars = project_to_time_grid(times, num_of_2_instars_in_time, time_grid_t)
    projected_num_of_2_outstars = project_to_time_grid(times, num_of_2_outstars_in_time, time_grid_t)

    return (projected_counts, projected_edge_type_counts, projected_tripoint_type_counts, projected_num_of_1_edge_causal, projected_num_of_2_chains, projected_num_of_2_instars, projected_num_of_2_outstars, current_time)

def run_realization_chunk(args):
    chunk_size, args_template = args
    results = []

    for i in range(chunk_size):
        results.append(run_one_realization(args_template))

    return results


def run_sim(N_instances,N_processes_per_instance,N_vertices_full,p_edges,n_states,n_edge_types,fractions_initial,
                     allowed_edges,inv_edge_rates,allowed_vertices,inv_vertex_rates,
                     time_grid_t,N_time_bins,T_max,
                     filename,graphs_dir,curves_dir,curves_dir_full,hrcak,
                     stability_checker,
                    parallel_mode='realizations', num_workers=None, nice_level=10,
                    inner_num_workers=1, chunk_size=10):

    cpu_total = multiprocessing.cpu_count()
    N_RESERVED_CORES = 2

    if cpu_total <= N_RESERVED_CORES:
        raise RuntimeError("Not enough CPU cores to reserve")

    allowed_cores = list(range(cpu_total - N_RESERVED_CORES))

    try:
        os.sched_setaffinity(0, allowed_cores)
        print(f"Main process pinned to cores: {allowed_cores}")
        print(f"Reserved free cores: {list(range(cpu_total - N_RESERVED_CORES, cpu_total))}")
    except Exception:
        pass

    # Default serial/realization-parallel behavior follows
    i_instance = 0
    n_valid_graphs = 0 # when there is not enough sources to infect in desired numbers, we discard that graph
    #******
    t_max_reached=[]
    #******

    while n_valid_graphs < N_instances:

        ###################
        # CONSTRUCT GRAPH #
        ###################

        v1_sorted_full_idx , v2_sorted_by_v1_full_idx = generate_undirected_ER_graph(
        N_vertices_full, p_edges
        )

        (N_vertices_in_LCC,
        v1_sorted,
        v2_sorted_by_v1) = take_largest_connected_component(
            N_vertices_full,
            v1_sorted_full_idx , v2_sorted_by_v1_full_idx
            )
        
        if N_vertices_in_LCC == 0:  # figure out what to expect and keep it ijn a few %
            i_instance += 1
            continue

        (v1_sorted_by_v2, v2_sorted,
        edge_ids_sorted_by_v2) = sort_v1_by_v2(v1_sorted, v2_sorted_by_v1)

        N_edges = len(v1_sorted)

        deg = np.zeros(N_vertices_in_LCC, dtype=np.int64)
    
        np.add.at(deg, v1_sorted, 1)
        np.add.at(deg, v2_sorted_by_v1, 1)

        N_2edges = np.sum(deg * (deg - 1) // 2)


        ptr_v1 = v1_pointer(v1_sorted, N_vertices_in_LCC)
        ptr_v2 = v2_pointer(v2_sorted, N_vertices_in_LCC)


        graph_path = graphs_dir / f"graph_ER_undirected_instanceNo{n_valid_graphs:04d}_N{N_vertices_full}_Nconnected{N_vertices_in_LCC}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
        check_disk_space(hrcak, min_free_GB=5)
        np.savez_compressed(
            graph_path,
            instance_number = n_valid_graphs,
            N_vertices_full=N_vertices_full,
            p_edges=p_edges,
            N_vertices_in_LCC = N_vertices_in_LCC,
            N_edges = N_edges,
            N_2edges = N_2edges,
            v1_sorted=v1_sorted,
            v2_sorted_by_v1=v2_sorted_by_v1
        )

        graph_path_csv = graphs_dir / f"graph_short_info_ER_undirected_instanceNo{n_valid_graphs:04d}_N{N_vertices_full}_Nconnected{N_vertices_in_LCC}_k{p_edges*(N_vertices_full-1)}_{filename}.csv"
        np.savetxt(
            graph_path_csv,
            [[
                n_valid_graphs,
                N_vertices_full,
                p_edges,
                N_vertices_in_LCC,
                N_edges,
                N_2edges
            ]],
            delimiter=",",
            header="instance_number,N_vertices_full,p_edges,N_vertices_in_LCC,N_edges,N_2edges",
            comments=""
        )
        ######################################
        # ZEROING THE IN_ONE_INSTANCE ARRAYS #
        ######################################

        mean_counts_in_time_in_one_instance = np.zeros((N_time_bins, n_states))
        M2_counts_in_time_in_one_instance = np.zeros((N_time_bins, n_states))

        mean_edge_type_counts_in_time_in_one_instance = np.zeros((N_time_bins, n_edge_types)) # (SS,SI,II)
        M2_edge_type_counts_in_time_in_one_instance = np.zeros((N_time_bins, n_edge_types))

        mean_tripoint_type_counts_in_time_in_one_instance = np.zeros((N_time_bins, 6)) # (SSS,SSI = ISS,ISI,SIS,SII = IIS ,III)
        M2_tripoint_type_counts_in_time_in_one_instance = np.zeros((N_time_bins, 6)) 

        mean_num_of_1_edge_causal_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_1_edge_causal_in_time_in_one_instance = np.zeros(N_time_bins)
        mean_num_of_2_chains_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_2_chains_in_time_in_one_instance = np.zeros(N_time_bins)
        mean_num_of_2_instars_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_2_instars_in_time_in_one_instance = np.zeros(N_time_bins)
        mean_num_of_2_outstars_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_2_outstars_in_time_in_one_instance = np.zeros(N_time_bins)    

        # CAUSAL MOTIFS ARE NONSENSE IN THIS FOR NOW

        # Parallel execution of processes (each realization is independent)
        if N_processes_per_instance <= 0:
            continue

        # num_workers=None → polite default (half cores)
        # num_workers=int  → explicit override

        usable_cores = len(allowed_cores)

        if num_workers is None:
            _n_workers = usable_cores - 2  #max(1, usable_cores // 2)   # polite default
        else:
            _n_workers = max(1, min(int(num_workers), usable_cores))




        # never spawn more workers than allowed cores
        _n_workers = min(_n_workers, len(allowed_cores))
        MAX_IN_FLIGHT = 2 * _n_workers

        print(f"Workers pinned to cores: {sorted(allowed_cores)}")
        print(f"Reserved free cores: {list(range(cpu_total - N_RESERVED_CORES, cpu_total))}")
        
        print(
            f"Scheduling {N_processes_per_instance} realizations "
            f"using {_n_workers} workers, chunk_size={chunk_size}, "
            f"MAX_IN_FLIGHT={MAX_IN_FLIGHT}"
        )
        
        args_template = (
                         N_vertices_full,
                         N_vertices_in_LCC,
                         v1_sorted,
                         v2_sorted_by_v1,
                         v1_sorted_by_v2,
                         edge_ids_sorted_by_v2,
                         deg, 
                         ptr_v1,
                         ptr_v2,
                         N_edges,
                         n_states,
                         n_edge_types,
                         fractions_initial,
                         allowed_edges,
                         inv_edge_rates,
                         allowed_vertices,
                         inv_vertex_rates,
                         time_grid_t,
                         N_time_bins,
                         T_max
                         )

        # reset running statistics (already zeroed above)
        processed = 0

        # Use ProcessPoolExecutor; workers lowered in priority in initializer
        try:
            with ProcessPoolExecutor(
                max_workers=_n_workers,
                initializer=_worker_initializer,
                initargs=(nice_level, allowed_cores, 0)
            ) as executor:

                futures = set()

                remaining = N_processes_per_instance

                while remaining > 0 or futures:

                    while remaining > 0 and len(futures) < MAX_IN_FLIGHT:
                        this_chunk = min(chunk_size, remaining)
                        remaining -= this_chunk

                        futures.add(
                            executor.submit(
                                run_realization_chunk,
                                (this_chunk, args_template)
                            )
                        )

                    done, futures = wait(futures, return_when=FIRST_COMPLETED)

                    for fut in done:
                        chunk_results = fut.result()

                        for (projected_counts,
                             projected_edge_type_counts,
                             projected_tripoint_type_counts,
                            projected_num_of_1_edge_causal,
                            projected_num_of_2_chains,
                            projected_num_of_2_instars,
                            projected_num_of_2_outstars,    
                            current_time) in chunk_results:

                            if processed >= N_processes_per_instance:
                                break

                            t_max_reached.append(current_time)

                            (mean_counts_in_time_in_one_instance,
                            M2_counts_in_time_in_one_instance) = update_online_mean_var(
                                projected_counts,
                                processed,
                                mean_counts_in_time_in_one_instance,
                                M2_counts_in_time_in_one_instance
                            )

                            (mean_edge_type_counts_in_time_in_one_instance,
                            M2_edge_type_counts_in_time_in_one_instance) = update_online_mean_var(
                                projected_edge_type_counts,
                                processed,
                                mean_edge_type_counts_in_time_in_one_instance,
                                M2_edge_type_counts_in_time_in_one_instance
                            )

                            (mean_tripoint_type_counts_in_time_in_one_instance,
                            M2_tripoint_type_counts_in_time_in_one_instance) = update_online_mean_var(
                                projected_tripoint_type_counts,
                                processed,
                                mean_tripoint_type_counts_in_time_in_one_instance,
                                M2_tripoint_type_counts_in_time_in_one_instance
                            )
                            

                            (mean_num_of_1_edge_causal_in_time_in_one_instance,
                            M2_num_of_1_edge_causal_in_time_in_one_instance) = update_online_mean_var(
                                projected_num_of_1_edge_causal,
                                processed,
                                mean_num_of_1_edge_causal_in_time_in_one_instance,
                                M2_num_of_1_edge_causal_in_time_in_one_instance
                            )

                            (mean_num_of_2_chains_in_time_in_one_instance,
                            M2_num_of_2_chains_in_time_in_one_instance) = update_online_mean_var(
                                projected_num_of_2_chains,
                                processed,
                                mean_num_of_2_chains_in_time_in_one_instance,
                                M2_num_of_2_chains_in_time_in_one_instance
                            )

                            (mean_num_of_2_instars_in_time_in_one_instance,
                            M2_num_of_2_instars_in_time_in_one_instance) = update_online_mean_var(
                                projected_num_of_2_instars,
                                processed,
                                mean_num_of_2_instars_in_time_in_one_instance,
                                M2_num_of_2_instars_in_time_in_one_instance
                            )

                            (mean_num_of_2_outstars_in_time_in_one_instance,
                            M2_num_of_2_outstars_in_time_in_one_instance) = update_online_mean_var(
                                projected_num_of_2_outstars,
                                processed,
                                mean_num_of_2_outstars_in_time_in_one_instance,
                                M2_num_of_2_outstars_in_time_in_one_instance
                            )

                            processed += 1

                            if processed % 100 == 0:
                                print(f"[instance {n_valid_graphs}] processed {processed}/{N_processes_per_instance}")

                            if processed in stability_checker:
                                curves_path = curves_dir / f"curves_instanceNo{n_valid_graphs:04d}_Nprocesses{processed}_N{N_vertices_full}_Nconnected{N_vertices_in_LCC}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
                                curves_path_full = curves_dir_full / f"curves_FULL_instanceNo{n_valid_graphs:04d}_Nprocesses{processed}_N{N_vertices_full}_Nconnected{N_vertices_in_LCC}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
                                check_disk_space(hrcak, min_free_GB=5)
                                np.savez_compressed(
                                    curves_path,
                                    instance_number=n_valid_graphs,
                                    N_vertices_full=N_vertices_full,
                                    p_edges=p_edges,
                                    N_vertices_in_LCC=N_vertices_in_LCC,
                                    N_processes=processed,
                                    time_grid=time_grid_t,
                                    mean_fractions=mean_counts_in_time_in_one_instance / N_vertices_in_LCC,
                                    var_fractions=(M2_counts_in_time_in_one_instance/ (N_vertices_in_LCC**2) / max(1, processed-1)),
                                    mean_edge_type_counts_in_time_in_one_instance=mean_edge_type_counts_in_time_in_one_instance/ N_edges,
                                    var_edge_type_counts=(M2_edge_type_counts_in_time_in_one_instance/ (N_edges**2) / max(1, processed-1)),
                                    mean_tripoint_type_counts_in_time_in_one_instance=mean_tripoint_type_counts_in_time_in_one_instance/ N_2edges,
                                    var_tripoint_type_counts=(M2_tripoint_type_counts_in_time_in_one_instance/ (N_2edges**2) / max(1, processed-1)),
                                    mean_num_of_1_edge_causal_in_time_in_one_instance=mean_num_of_1_edge_causal_in_time_in_one_instance/ N_vertices_in_LCC,
                                    var_num_of_1_edge_causal_in_time_in_one_instance=M2_num_of_1_edge_causal_in_time_in_one_instance/ (N_vertices_in_LCC**2) / max(1, processed-1),
                                    mean_num_of_2_chains_in_time_in_one_instance=mean_num_of_2_chains_in_time_in_one_instance/ N_vertices_in_LCC,
                                    var_num_of_2_chains_in_time_in_one_instance=M2_num_of_2_chains_in_time_in_one_instance/ (N_vertices_in_LCC**2) / max(1, processed-1),
                                    mean_num_of_2_instars_in_time_in_one_instance=mean_num_of_2_instars_in_time_in_one_instance/ N_vertices_in_LCC,
                                    var_num_of_2_instars_in_time_in_one_instance=M2_num_of_2_instars_in_time_in_one_instance/ (N_vertices_in_LCC**2) / max(1, processed-1),
                                    mean_num_of_2_outstars_in_time_in_one_instance=mean_num_of_2_outstars_in_time_in_one_instance/ N_vertices_in_LCC,
                                    var_num_of_2_outstars_in_time_in_one_instance=M2_num_of_2_outstars_in_time_in_one_instance/ (N_vertices_in_LCC**2) / max(1, processed-1),
                                )



        except Exception as e:
            print(f"Parallel execution failed ({e}), falling back to serial loop")

            for i_process in range(N_processes_per_instance):

                # =====================================================
                # INITIAL CONDITIONS (consistent with main pipeline)
                # =====================================================
                vertex_states = initial_vertex_states_SI(
                    N_vertices_in_LCC,
                    np.arange(N_vertices_in_LCC),
                    fractions_initial
                )

                edge_types = find_edge_types(
                    v1_sorted,
                    v2_sorted_by_v1,
                    vertex_states,
                    n_states
                )

                edge_events = init_edge_events(edge_types, allowed_edges, inv_edge_rates)
                vertex_events = init_vertex_events(vertex_states, allowed_vertices, inv_vertex_rates, N_edges)

                events = np.vstack((edge_events, vertex_events))

                causal = np.zeros(N_edges, dtype=bool)
                causal_in = np.zeros(N_vertices_in_LCC, dtype=int)
                causal_out = np.zeros(N_vertices_in_LCC, dtype=int)

                m = np.zeros(N_vertices_in_LCC, dtype=np.int64)

                m[v1_sorted] += vertex_states[v2_sorted_by_v1]
                m[v2_sorted_by_v1] += vertex_states[v1_sorted]

                S = (vertex_states == 0)
                I = (vertex_states == 1)

                degmm = deg - m

                deg_s = deg[S]
                m_s = m[S]
                degmm_s = degmm[S]

                deg_i = deg[I]
                m_i = m[I]
                degmm_i = degmm[I]


                # =====================================================
                # SKM INITIALIZATION (dense, vectorized)
                # =====================================================

                k_max = deg.max()

                # Dense matrix: rows = k, cols = m
                skm = np.zeros((k_max + 1, k_max + 1), dtype=np.int64)

                # Accumulate counts
                np.add.at(skm, (deg_s, m_s), 1)

                # =====================================================
                # TRACKING
                # =====================================================
                times = []
                counts_in_time = []
                edge_type_counts_in_time = []
                tripoint_type_counts_in_time = []

                num_new_1_edge_causal = 0
                num_new_2_chains = 0
                num_new_2_instars = 0
                num_new_2_outstars = 0

                total_1_edge_causal = 0
                total_2_chains = 0
                total_2_instars = 0
                total_2_outstars = 0

                num_of_1_edge_causal_in_time = []
                num_of_2_chains_in_time = []
                num_of_2_instars_in_time = []
                num_of_2_outstars_in_time = []

                current_time = 0.0
                current_counts = np.bincount(vertex_states, minlength=n_states)
                edge_type_current_counts = np.bincount(edge_types, minlength=n_edge_types) # (SS,SI,II)
                tripoint_type_current_counts = np.zeros(6, dtype=int) # (SSS,SSI = ISS,ISI,SIS,SII = IIS ,III) -  ignore for now
                check = 1

                # =====================================================
                # MAIN LOOP (UNDIRECTED step)
                # =====================================================
                while current_time < T_max and check:

                    times.append(current_time)
                    counts_in_time.append(current_counts)
                    edge_type_counts_in_time.append(edge_type_current_counts)
                    tripoint_type_counts_in_time.append(tripoint_type_current_counts)

                    total_1_edge_causal += num_new_1_edge_causal
                    total_2_chains += num_new_2_chains
                    total_2_instars += num_new_2_instars
                    total_2_outstars += num_new_2_outstars

                    num_of_1_edge_causal_in_time.append(total_1_edge_causal)
                    num_of_2_chains_in_time.append(total_2_chains)
                    num_of_2_instars_in_time.append(total_2_instars)
                    num_of_2_outstars_in_time.append(total_2_outstars)

                    

                    (check,
                    current_time,
                    current_counts,
                    num_new_1_edge_causal,
                    num_new_2_chains,
                    num_new_2_instars,
                    num_new_2_outstars,
                    edge_type_current_counts,
                    tripoint_type_current_counts,
                    m, S, I, deg_s,deg_i, m_s, m_i, degmm_s, degmm_i,skm) = step(
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
                    )

                # =====================================================
                # POST-PROCESSING
                # =====================================================
                t_max_reached.append(current_time)

                projected_counts = project_to_time_grid(times, counts_in_time, time_grid_t)

                (mean_counts_in_time_in_one_instance,
                M2_counts_in_time_in_one_instance) = update_online_mean_var(
                    projected_counts,
                    i_process,
                    mean_counts_in_time_in_one_instance,
                    M2_counts_in_time_in_one_instance
                )


                projected_edge_type_counts = project_to_time_grid(times, edge_type_counts_in_time, time_grid_t)

                (mean_edge_type_counts_in_time_in_one_instance,
                M2_edge_type_counts_in_time_in_one_instance) = update_online_mean_var(
                    projected_edge_type_counts,
                    processed,
                    mean_edge_type_counts_in_time_in_one_instance,
                    M2_edge_type_counts_in_time_in_one_instance
                )

                projected_tripoint_type_counts = project_to_time_grid(times, tripoint_type_counts_in_time, time_grid_t)

                (mean_tripoint_type_counts_in_time_in_one_instance,
                M2_tripoint_type_counts_in_time_in_one_instance) = update_online_mean_var(
                    projected_tripoint_type_counts,
                    processed,
                    mean_tripoint_type_counts_in_time_in_one_instance,
                    M2_tripoint_type_counts_in_time_in_one_instance
                )

                projected_num_of_1_edge_causal = project_to_time_grid(
                    times, num_of_1_edge_causal_in_time, time_grid_t
                )

                (mean_num_of_1_edge_causal_in_time_in_one_instance,
                M2_num_of_1_edge_causal_in_time_in_one_instance) = update_online_mean_var(
                    projected_num_of_1_edge_causal,
                    i_process,
                    mean_num_of_1_edge_causal_in_time_in_one_instance,
                    M2_num_of_1_edge_causal_in_time_in_one_instance
                )

                projected_num_of_2_chains = project_to_time_grid(
                    times, num_of_2_chains_in_time, time_grid_t
                )

                (mean_num_of_2_chains_in_time_in_one_instance,
                M2_num_of_2_chains_in_time_in_one_instance) = update_online_mean_var(
                    projected_num_of_2_chains,
                    i_process,
                    mean_num_of_2_chains_in_time_in_one_instance,
                    M2_num_of_2_chains_in_time_in_one_instance
                )

                projected_num_of_2_instars = project_to_time_grid(
                    times, num_of_2_instars_in_time, time_grid_t
                )

                (mean_num_of_2_instars_in_time_in_one_instance,
                M2_num_of_2_instars_in_time_in_one_instance) = update_online_mean_var(
                    projected_num_of_2_instars,
                    i_process,
                    mean_num_of_2_instars_in_time_in_one_instance,
                    M2_num_of_2_instars_in_time_in_one_instance
                )

                projected_num_of_2_outstars = project_to_time_grid(
                    times, num_of_2_outstars_in_time, time_grid_t
                )

                (mean_num_of_2_outstars_in_time_in_one_instance,
                M2_num_of_2_outstars_in_time_in_one_instance) = update_online_mean_var(
                    projected_num_of_2_outstars,
                    i_process,
                    mean_num_of_2_outstars_in_time_in_one_instance,
                    M2_num_of_2_outstars_in_time_in_one_instance
                )

                if i_process + 1 in stability_checker:
                    curves_path = curves_dir / f"curves_instanceNo{n_valid_graphs:04d}_Nprocesses{i_process+1}_N{N_vertices_full}_Nconnected{N_vertices_in_LCC}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
                    curves_path_full = curves_dir_full / f"curves_FULL_instanceNo{n_valid_graphs:04d}_Nprocesses{i_process+1}_N{N_vertices_full}_Nconnected{N_vertices_in_LCC}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
                    check_disk_space(hrcak, min_free_GB=5)

                    np.savez_compressed(
                        curves_path,
                        instance_number=n_valid_graphs,
                        N_vertices_full=N_vertices_full,
                        p_edges=p_edges,
                        N_vertices_in_LCC=N_vertices_in_LCC,
                        N_processes=i_process + 1,
                        time_grid=time_grid_t,

                        mean_fractions=mean_counts_in_time_in_one_instance / N_vertices_in_LCC,
                        var_fractions=(M2_counts_in_time_in_one_instance / (N_vertices_in_LCC**2) / max(1, i_process)),

                        mean_edge_type_counts_in_time_in_one_instance=mean_edge_type_counts_in_time_in_one_instance / N_edges,
                        var_edge_type_counts=(M2_edge_type_counts_in_time_in_one_instance / (N_edges**2) / max(1, i_process)),

                        mean_tripoint_type_counts_in_time_in_one_instance=mean_tripoint_type_counts_in_time_in_one_instance / N_2edges,
                        var_tripoint_type_counts=(M2_tripoint_type_counts_in_time_in_one_instance / (N_2edges**2) / max(1, i_process)),

                        mean_num_of_1_edge_causal_in_time_in_one_instance=mean_num_of_1_edge_causal_in_time_in_one_instance / N_vertices_in_LCC,
                        var_num_of_1_edge_causal_in_time_in_one_instance=M2_num_of_1_edge_causal_in_time_in_one_instance / (N_vertices_in_LCC**2) / max(1, i_process),

                        mean_num_of_2_chains_in_time_in_one_instance=mean_num_of_2_chains_in_time_in_one_instance / N_vertices_in_LCC,
                        var_num_of_2_chains_in_time_in_one_instance=M2_num_of_2_chains_in_time_in_one_instance / (N_vertices_in_LCC**2) / max(1, i_process),

                        mean_num_of_2_instars_in_time_in_one_instance=mean_num_of_2_instars_in_time_in_one_instance / N_vertices_in_LCC,
                        var_num_of_2_instars_in_time_in_one_instance=M2_num_of_2_instars_in_time_in_one_instance / (N_vertices_in_LCC**2) / max(1, i_process),

                        mean_num_of_2_outstars_in_time_in_one_instance=mean_num_of_2_outstars_in_time_in_one_instance / N_vertices_in_LCC,
                        var_num_of_2_outstars_in_time_in_one_instance=M2_num_of_2_outstars_in_time_in_one_instance / (N_vertices_in_LCC**2) / max(1, i_process),
                    )

                    np.savez_compressed(
                        curves_path_full,
                        instance_number=n_valid_graphs,
                        N_vertices_full=N_vertices_full,
                        p_edges=p_edges,
                        N_vertices_in_LCC=N_vertices_full,
                        N_processes=i_process + 1,
                        time_grid=time_grid_t,
                        mean_fractions_full=mean_counts_in_time_in_one_instance / N_vertices_full,
                        var_fractions_full=(M2_counts_in_time_in_one_instance / (N_vertices_full**2) / max(1, i_process)),

                        mean_num_of_1_edge_causal_in_time_in_one_instance_full=mean_num_of_1_edge_causal_in_time_in_one_instance / N_vertices_full,
                        var_num_of_1_edge_causal_in_time_in_one_instance_full=M2_num_of_1_edge_causal_in_time_in_one_instance / (N_vertices_full**2) / max(1, i_process),

                        mean_num_of_2_chains_in_time_in_one_instance_full=mean_num_of_2_chains_in_time_in_one_instance / N_vertices_full,
                        var_num_of_2_chains_in_time_in_one_instance_full=M2_num_of_2_chains_in_time_in_one_instance / (N_vertices_full**2) / max(1, i_process),

                        mean_num_of_2_instars_in_time_in_one_instance_full=mean_num_of_2_instars_in_time_in_one_instance / N_vertices_full,
                        var_num_of_2_instars_in_time_in_one_instance_full=M2_num_of_2_instars_in_time_in_one_instance / (N_vertices_full**2) / max(1, i_process),

                        mean_num_of_2_outstars_in_time_in_one_instance_full=mean_num_of_2_outstars_in_time_in_one_instance / N_vertices_full,
                        var_num_of_2_outstars_in_time_in_one_instance_full=M2_num_of_2_outstars_in_time_in_one_instance / (N_vertices_full**2) / max(1, i_process),
                    )

        n_valid_graphs += 1
        i_instance += 1
        

    t_max_reached = np.asarray(t_max_reached)
    N_realizations = N_instances*N_processes_per_instance
    times_path = curves_dir / f"times_check_N{N_vertices_full}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
    check_disk_space(hrcak, min_free_GB=5)
    np.savez_compressed(
        times_path,
        N_vertices_full=N_vertices_full,
        p_edges=p_edges,
        N_realizations=N_realizations,
        t_max=t_max_reached.max(),
        t_min=t_max_reached.min(),
        t_mean=t_max_reached.mean(),
        t_var=t_max_reached.var(),
        t_max_reached=t_max_reached
    )

