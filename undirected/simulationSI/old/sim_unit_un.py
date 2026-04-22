import numpy as np
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from graph_undirected import *
from sim_step_un import *
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
    (N_vertices_full,N_vertices, sources, sources_sorted, targets_sorted_by_source, sources_sorted_by_target, targets_sorted,
     edge_ids_sorted_by_target, ptr_src, ptr_tgt, N_edges, n_states,
     fractions_initial, allowed_edges, inv_edge_rates, allowed_vertices, inv_vertex_rates,
     time_grid_t, N_time_bins, T_max) = args
    
    print("\n\nNEW REALIZATION STARTED\n")
    print("\nThis realization is running on the following instance:\n")
    graph_show = np.column_stack((sources_sorted, targets_sorted_by_source))
    print("\n",graph_show,"\n")



    # INITIALIZE STATES
    vertex_states = initial_vertex_states_SI(N_vertices, sources, fractions_initial)
    edge_types = find_edge_types(sources_sorted,targets_sorted_by_source,vertex_states,n_states)
    edge_events = init_edge_events(edge_types, allowed_edges, inv_edge_rates)
    vertex_events = init_vertex_events(vertex_states, allowed_vertices, inv_vertex_rates, N_edges)

    events = np.vstack((edge_events, vertex_events))

    # SI SPECIFIC
    n_infected_init = sum(vertex_states)

    causal = np.zeros(N_edges, dtype=bool)
    causal_in  = np.zeros(N_vertices, dtype=int)
    causal_out = np.zeros(N_vertices, dtype=int)

    times = []
    counts_in_time = []

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
    check = 1

    while current_time < T_max and check:
        times.append(current_time)
        counts_in_time.append(current_counts)

        total_1_edge_causal += num_new_1_edge_causal
        total_2_chains += num_new_2_chains
        total_2_instars += num_new_2_instars
        total_2_outstars += num_new_2_outstars
        num_of_1_edge_causal_in_time.append(total_1_edge_causal)
        num_of_2_chains_in_time.append(total_2_chains)
        num_of_2_instars_in_time.append(total_2_instars)
        num_of_2_outstars_in_time.append(total_2_outstars)

        print("t: ",current_time)
        print("vertex states:", vertex_states)

        (check,
         current_time,
         current_counts,
         num_new_1_edge_causal,
         num_new_2_chains,
         num_new_2_instars,
         num_new_2_outstars) = step(
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
                                        )

    projected_counts = project_to_time_grid(times, counts_in_time, time_grid_t)
    projected_num_of_1_edge_causal = project_to_time_grid(times, num_of_1_edge_causal_in_time, time_grid_t)
    projected_num_of_2_chains = project_to_time_grid(times, num_of_2_chains_in_time, time_grid_t)
    projected_num_of_2_instars = project_to_time_grid(times, num_of_2_instars_in_time, time_grid_t)
    projected_num_of_2_outstars = project_to_time_grid(times, num_of_2_outstars_in_time, time_grid_t)

    return (projected_counts, projected_num_of_1_edge_causal, projected_num_of_2_chains, projected_num_of_2_instars, projected_num_of_2_outstars, current_time)

def run_realization_chunk(args):
    chunk_size, args_template = args
    results = []

    for i in range(chunk_size):
        results.append(run_one_realization(args_template))

    return results


def run_sim(N_instances,N_processes_per_instance,N_vertices_full,p_edges,n_states,fractions_initial,
                     allowed_edges,inv_edge_rates,allowed_vertices,inv_vertex_rates,
                     time_grid_t,N_time_bins,T_max,
                     filename,graphs_dir,curves_dir,hrcak,
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
        #print(f"Main process pinned to cores: {allowed_cores}")
        #print(f"Reserved free cores: {list(range(cpu_total - N_RESERVED_CORES, cpu_total))}")
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

        sources_sorted_full_idx, targets_sorted_by_source_full_idx = generate_directed_ER_graph(
        N_vertices_full, p_edges
        )

        (N_vertices,
        sources_sorted,
        targets_sorted_by_source) = take_largest_weakly_connected_component(
            N_vertices_full,
            sources_sorted_full_idx,
            targets_sorted_by_source_full_idx
            )
        
        if N_vertices == 0:  # figure out what to expect and keep it ijn a few %
            i_instance += 1
            continue

        (sources_sorted_by_target, targets_sorted,
        edge_ids_sorted_by_target) = sort_by_target(sources_sorted, targets_sorted_by_source)

        N_edges = len(sources_sorted)

        ptr_src = source_pointer(sources_sorted, N_vertices)
        ptr_tgt = target_pointer(targets_sorted, N_vertices)

        sources = np.nonzero(ptr_src[1:] - ptr_src[:-1])[0]
        targets = np.nonzero(ptr_tgt[1:] - ptr_tgt[:-1])[0]

        counts_nonzero_initial = np.round(fractions_initial)
        N_sources = len(sources)
        
        if counts_nonzero_initial.sum() > N_sources: # more seeds than available sources
            i_instance += 1
            continue

        graph_path = graphs_dir / f"graph_ER_directed_instanceNo{n_valid_graphs:04d}_N{N_vertices_full}_Nconnected{N_vertices}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
        check_disk_space(hrcak, min_free_GB=5)
        np.savez_compressed(
            graph_path,
            instance_number = n_valid_graphs,
            N_vertices_full=N_vertices_full,
            p_edges=p_edges,
            N_vertices_connected = N_vertices,
            sources=sources_sorted,
            targets=targets_sorted_by_source
        )

        ######################################
        # ZEROING THE IN_ONE_INSTANCE ARRAYS #
        ######################################

        mean_counts_in_time_in_one_instance = np.zeros((N_time_bins, n_states))
        M2_counts_in_time_in_one_instance = np.zeros((N_time_bins, n_states))

        mean_num_of_1_edge_causal_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_1_edge_causal_in_time_in_one_instance = np.zeros(N_time_bins)
        mean_num_of_2_chains_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_2_chains_in_time_in_one_instance = np.zeros(N_time_bins)
        mean_num_of_2_instars_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_2_instars_in_time_in_one_instance = np.zeros(N_time_bins)
        mean_num_of_2_outstars_in_time_in_one_instance = np.zeros(N_time_bins)
        M2_num_of_2_outstars_in_time_in_one_instance = np.zeros(N_time_bins)    

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

        #print(f"Workers pinned to cores: {sorted(allowed_cores)}")
        #print(f"Reserved free cores: {list(range(cpu_total - N_RESERVED_CORES, cpu_total))}")
        '''
        print(
            f"Scheduling {N_processes_per_instance} realizations "
            f"using {_n_workers} workers, chunk_size={chunk_size}, "
            f"MAX_IN_FLIGHT={MAX_IN_FLIGHT}"
        )
        '''
        args_template = (
                         N_vertices_full,
                         N_vertices,
                         sources,
                         sources_sorted,
                         targets_sorted_by_source,
                         sources_sorted_by_target,
                         targets_sorted,
                         edge_ids_sorted_by_target,
                         ptr_src,
                         ptr_tgt,
                         N_edges,
                         n_states,
                         fractions_initial,
                         allowed_edges,
                         inv_edge_rates,
                         allowed_vertices,
                         inv_vertex_rates,
                         time_grid_t,
                         N_time_bins,
                         T_max)

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
                                curves_path = curves_dir / f"curves_instanceNo{n_valid_graphs:04d}_Nprocesses{processed}_N{N_vertices_full}_Nconnected{N_vertices}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
                                check_disk_space(hrcak, min_free_GB=5)
                                np.savez_compressed(
                                    curves_path,
                                    instance_number=n_valid_graphs,
                                    N_vertices_full=N_vertices_full,
                                    p_edges=p_edges,
                                    N_vertices_connected=N_vertices,
                                    N_processes=processed,
                                    time_grid=time_grid_t,
                                    mean_fractions=mean_counts_in_time_in_one_instance / N_vertices,
                                    var_fractions=(M2_counts_in_time_in_one_instance/ (N_vertices**2) / max(1, processed-1)),
                                    mean_num_of_1_edge_causal_in_time_in_one_instance=mean_num_of_1_edge_causal_in_time_in_one_instance/ N_vertices,
                                    var_num_of_1_edge_causal_in_time_in_one_instance=M2_num_of_1_edge_causal_in_time_in_one_instance/ (N_vertices**2) / max(1, processed-1),
                                    mean_num_of_2_chains_in_time_in_one_instance=mean_num_of_2_chains_in_time_in_one_instance/ N_vertices,
                                    var_num_of_2_chains_in_time_in_one_instance=M2_num_of_2_chains_in_time_in_one_instance/ (N_vertices**2) / max(1, processed-1),
                                    mean_num_of_2_instars_in_time_in_one_instance=mean_num_of_2_instars_in_time_in_one_instance/ N_vertices,
                                    var_num_of_2_instars_in_time_in_one_instance=M2_num_of_2_instars_in_time_in_one_instance/ (N_vertices**2) / max(1, processed-1),
                                    mean_num_of_2_outstars_in_time_in_one_instance=mean_num_of_2_outstars_in_time_in_one_instance/ N_vertices,
                                    var_num_of_2_outstars_in_time_in_one_instance=M2_num_of_2_outstars_in_time_in_one_instance/ (N_vertices**2) / max(1, processed-1),
                                )



        except Exception as e:
            # Fall back to serial execution if parallel fails
            print(f"Parallel execution failed ({e}), falling back to serial loop")
            for i_process in range(N_processes_per_instance):
                vertex_states = initial_vertex_states_SI(N_vertices, sources, fractions_initial) # before it was N_vertices
                edge_types = find_edge_types(sources_sorted,targets_sorted_by_source,vertex_states,n_states)
                edge_events = init_edge_events(edge_types, allowed_edges, inv_edge_rates)
                vertex_events = init_vertex_events(vertex_states, allowed_vertices, inv_vertex_rates, N_edges)

                events = np.vstack((edge_events, vertex_events))

                n_infected_init = sum(vertex_states)

                causal = np.zeros(N_edges, dtype=bool)
                causal_in  = np.zeros(N_vertices, dtype=int)
                causal_out = np.zeros(N_vertices, dtype=int)

                times = []
                counts_in_time = []

                num_new_1_edge_causal = 0
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
                check = 1

                while current_time<T_max and check:
                    times.append(current_time)
                    counts_in_time.append(current_counts)

                    total_1_edge_causal += num_new_1_edge_causal
                    total_2_chains += num_new_2_chains
                    total_2_instars += num_new_2_instars
                    total_2_outstars += num_new_2_outstars
                    num_of_1_edge_causal_in_time.append(total_1_edge_causal)
                    num_of_2_chains_in_time.append(total_2_chains)
                    num_of_2_instars_in_time.append(total_2_instars)
                    num_of_2_outstars_in_time.append(total_2_outstars)

                    print("t: ",current_time)

                    (check, 
                    current_time, 
                    current_counts,
                    num_new_1_edge_causal,
                    num_new_2_chains,
                    num_new_2_instars,
                    num_new_2_outstars) = step(
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
                                            )

                t_max_reached.append(current_time)
                print("\n")

                projected_counts = project_to_time_grid(times, counts_in_time, time_grid_t)

                (mean_counts_in_time_in_one_instance,
                M2_counts_in_time_in_one_instance) = update_online_mean_var(
                                                                projected_counts,
                                                                i_process,
                                                                mean_counts_in_time_in_one_instance,
                                                                M2_counts_in_time_in_one_instance
                                                                    )

                var_counts_in_time_in_one_instance = M2_counts_in_time_in_one_instance / max(1, i_process)
                mean_fractions_in_time_in_one_instance = mean_counts_in_time_in_one_instance / N_vertices
                var_fractions_in_time_in_one_instance = var_counts_in_time_in_one_instance / (N_vertices ** 2)

                projected_num_of_1_edge_causal = project_to_time_grid(times, num_of_1_edge_causal_in_time,time_grid_t)

                (mean_num_of_1_edge_causal_in_time_in_one_instance,
                M2_num_of_1_edge_causal_in_time_in_one_instance) = update_online_mean_var(
                                                                                        projected_num_of_1_edge_causal,
                                                                                        i_process,
                                                                                        mean_num_of_1_edge_causal_in_time_in_one_instance,
                                                                                        M2_num_of_1_edge_causal_in_time_in_one_instance
                                                                                                )
                
                var_num_of_1_edge_causal_in_time_in_one_instance = M2_num_of_1_edge_causal_in_time_in_one_instance / max(1, i_process)

        

                projected_num_of_2_chains = project_to_time_grid(times, num_of_2_chains_in_time,time_grid_t)
                
                var_num_of_2_chains_in_time_in_one_instance = M2_num_of_2_chains_in_time_in_one_instance / max(1, i_process)

                projected_num_of_2_instars = project_to_time_grid(times, num_of_2_instars_in_time,time_grid_t)
                
                var_num_of_2_instars_in_time_in_one_instance = M2_num_of_2_instars_in_time_in_one_instance / max(1, i_process)

                projected_num_of_2_outstars = project_to_time_grid(times, num_of_2_outstars_in_time,time_grid_t)
                
                var_num_of_2_outstars_in_time_in_one_instance = M2_num_of_2_outstars_in_time_in_one_instance / max(1, i_process)

                if i_process + 1 in stability_checker:
                    curves_path = curves_dir / f"curves_instanceNo{n_valid_graphs:04d}_Nprocesses{i_process+1}_N{N_vertices_full}_Nconnected{N_vertices}_k{p_edges*(N_vertices_full-1)}_{filename}.npz"
                    check_disk_space(hrcak, min_free_GB=5)
                    np.savez_compressed(
                        curves_path,
                        instance_number = n_valid_graphs,
                        N_vertices_full=N_vertices_full,
                        p_edges=p_edges,
                        N_vertices_connected = N_vertices,
                        N_processes = i_process+1,
                        time_grid=time_grid_t,
                        mean_fractions=mean_fractions_in_time_in_one_instance,
                        var_fractions=var_fractions_in_time_in_one_instance,
                        mean_num_of_1_edge_causal_in_time_in_one_instance = mean_num_of_1_edge_causal_in_time_in_one_instance,
                        var_num_of_1_edge_causal_in_time_in_one_instance = var_num_of_1_edge_causal_in_time_in_one_instance,
                        mean_num_of_2_chains_in_time_in_one_instance = mean_num_of_2_chains_in_time_in_one_instance,
                        var_num_of_2_chains_in_time_in_one_instance = var_num_of_2_chains_in_time_in_one_instance,
                        mean_num_of_2_instars_in_time_in_one_instance = mean_num_of_2_instars_in_time_in_one_instance,
                        var_num_of_2_instars_in_time_in_one_instance = var_num_of_2_instars_in_time_in_one_instance,
                        mean_num_of_2_outstars_in_time_in_one_instance = mean_num_of_2_outstars_in_time_in_one_instance,
                        var_num_of_2_outstars_in_time_in_one_instance = var_num_of_2_outstars_in_time_in_one_instance
                    )

        n_valid_graphs += 1
        i_instance += 1
        print("\n")

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

