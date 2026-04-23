# current

import numpy as np

def make_time_grid(N, k, lambda_rate=1.0, safety=3.0, fine_factor=2.0):
    #sparse_limit = np.log(N) / np.log(max(k, 1.01))      # sparse regime
    #dense_limit  = np.log(N) / max(k, 0.1)              # dense regime
    #T_base = min(sparse_limit, dense_limit)
    T_max = 50 #safety * T_base / lambda_rate

    delta_t = 1 / (k * lambda_rate * fine_factor)

    N_time_bins = int(np.ceil(T_max / delta_t))

    t_grid = np.linspace(0, T_max, N_time_bins)
    return T_max, N_time_bins, t_grid

def project_to_time_grid(times, counts_in_time, time_grid):
    times = np.asarray(times)
    counts_in_time = np.asarray(counts_in_time)

    # searchsorted = at which index should each value be
    # inserted into a sorted array to keep it sorted?
    idx = np.searchsorted(times, time_grid, side="right") - 1
    # -1 to get the state after the most recent event but before the next grid time
    idx[idx < 0] = 0 # sorting out grid times before the first event

    return counts_in_time[idx] 
    # gives back only the states that happened last before the next grid time

def update_online_mean_var(projected, i_process, mean_old, M2_old): # i_process goes from 0! so +1
    delta = projected - mean_old
    mean_new = mean_old + delta / (i_process + 1)
    delta2 = projected - mean_new
    M2_new = M2_old + delta * delta2
    return mean_new, M2_new


