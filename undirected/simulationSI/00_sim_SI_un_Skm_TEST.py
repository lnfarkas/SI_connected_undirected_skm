# current

import numpy as np
import time

##########
# IMPORT #
##########

from graph_un import *
from step_un import *
from time_grid import *
from sim_unit_un_Skm_TEST import *
#from sim_unit_random_regular import * # this is the only difference in code from the ER version, the rest is the same

########
# PATH #
########

from pathlib import Path
from datetime import datetime

hrcak = Path("/home/lnf/Desktop")

today_str = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"SIsimUNDIRECTED{today_str}"
base_outdir = hrcak /f"00_sim_SI" / filename


################
# PARAMETERS 1 #
################

CHUNK_SIZE = 100   # paralelization chunk

N_instances = 2 #100

N_v = [20] #10000

#multiplier = 10 # N_processes_per_instance = multiplier * N_vertices_full

ks =  [2] # [2,5,10,20,50,100,200,500,1000,2000,5000,10000]

################
# PARAMETERS 2 #
################

n_states = 2

fractions_initial = np.array([0.1])  # fraction for state 1; state 0 inferred

edge_rates=np.array([-1.0,  1.0, -1.0]) # must be a NumPy array
allowed_edges, inv_edge_rates = prep_rates(edge_rates)

vertex_rates = np.array([-1.0, -1.0]) # must be a NumPy array
allowed_vertices, inv_vertex_rates = prep_rates(vertex_rates) # inv_rates = 1/rates, for the rnd exp generator

######################################################################################
# LOOP
######################################################################################


for N_vertices_full in N_v:

    for k in ks:

        if k<=N_vertices_full:

            p_edges = k / (N_vertices_full - 1)

            N_processes_per_instance = 2
            #N_processes_per_instance = multiplier*N_vertices_full

            T_max, N_time_bins, time_grid = make_time_grid(N_vertices_full, k, edge_rates[1], 7, 5)

            stability_checker = [1,2,3,4,5,6,7,8,9,10]
            #stability_checker = [N_vertices_full * i for i in range(1, multiplier+1)]

            spec_outdir = base_outdir / f"N{N_vertices_full}_k{k}"

            graphs_dir = spec_outdir / "Graphs"
            curves_dir = spec_outdir / "Curves"
            curves_dir_full = spec_outdir / "Curves_FULL"

            graphs_dir.mkdir(parents=True, exist_ok=True)
            curves_dir.mkdir(parents=True, exist_ok=True)
            curves_dir_full.mkdir(parents=True, exist_ok=True)


            start = time.perf_counter()

            # Run instances in parallel across CPU cores (keeps PC usable by default)
            num_workers = 15 #None  # set to an int to override, else uses cpu_count() - 1
            nice_level = 10     # niceness for worker processes (higher = lower priority)
            run_sim(N_instances,N_processes_per_instance,N_vertices_full,p_edges,n_states,fractions_initial,
                                allowed_edges,inv_edge_rates,allowed_vertices,inv_vertex_rates,
                                time_grid,N_time_bins,T_max,
                                filename,graphs_dir,curves_dir,curves_dir_full,hrcak,
                                stability_checker,
                                parallel_mode='realizations', num_workers=num_workers,
                                nice_level=nice_level, chunk_size=CHUNK_SIZE)

            N_realizations = N_instances*N_processes_per_instance

            end = time.perf_counter()
            elapsed_time = end - start
            # print(f"N_vertices (not necesarrily connected) = {N_vertices_full}, p_edges = {p_edges} (<k_in/out> = {(N_vertices_full-1)*p_edges})\n{N_instances} instances, {N_realizations} realizations ({N_processes_per_instance} per instance)\nElapsed time: {elapsed_time:.6f} seconds (wall-clock)\n{elapsed_time/N_realizations:.6f} seconds per realization")


