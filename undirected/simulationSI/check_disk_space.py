import psutil
import sys
import numpy as np

min_free_GB = 5

def check_disk_space(path='/', min_free_GB=5):
    free_bytes = psutil.disk_usage(path).free
    if free_bytes < min_free_GB * 1e9:
        #print(f"Not enough free space! Only {free_bytes/1e9:.2f} GB left. Stopping simulation.")
        sys.exit()
    #else:
        #print(f"Enough free space: {free_bytes/1e9:.2f} GB left. Continuing simulation.")