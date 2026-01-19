#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 13:48:48 2025

@author: jacopo.rodeschini@unibg.it
"""

from utils import task
import pickle
import gc
from itertools import product

import numpy as np
import jax
from joblib import Parallel, delayed
from pathlib import Path


# %% [Joblib paralled] Simulations

def main(folder_path):

    nvar = 3
    nlat = 2
    domain = np.array([1, 1, 1])  # [0,1]^2

    beta = np.array([1, 2, -1])  # b0 (nvar=0), b0 (nvar=1)
    s2 = 1                          # marginal variance matern
    ks = np.sqrt(8)*np.array([7, 2])       # range = 0.35
    nu = np.ones((nlat, 1))  # nu

    A = np.array([[0.5, 1], [0.5, 0.3], [0.2, 0.8]])
    flatent = [0.8,  -0.5]  # sorted in descending order
    s2error = np.array([0.5, 1.5, 1])  # measurement error

    T = [25, 50]  # T = [100, 200]
    mtotal = [25, 25, 25]  # n points = mtot**2
    mtrain = [[10, 10, 10], [15, 15, 15], [20, 20, 20]]  #

    # Random sample size
    lowrank = [1, 0.75, 0.5, 0.25, 0.15]  # Degree of low-rank [None = Full]
    boot = 100

    # Create all combinations: (l, m, t, seed) [cartesian product]
    param = list(product(mtrain, T))
    param = sorted(param, key=lambda x: x[1])

    main_key = jax.random.PRNGKey(1234)
    sub_key = jax.random.split(main_key, num=len(param))

    inx = 0
    for m, t in param:

        loop_key, main_key = jax.random.split(sub_key[inx, :])
        inx += 1

        for l in lowrank:
            loop_key, main_key = jax.random.split(main_key)

            # Create a BATCH of keys, one for each of the `boot` runs
            keys_for_pool = jax.random.split(loop_key, num=boot)

            print(f"Processing combination m={m}, t={t}, l={l}")
            results_list = Parallel(n_jobs=25, backend='loky')(
                delayed(task)(key, domain=domain, mtotal=mtotal, mtrain=m, A=A, ks=ks, beta=beta, T=t, s2error=s2error,
                              flatent=flatent, lowrank=l, max_iter=200, tol_relat=1e-4, verbose=False, estimates=False, regular=False) for key in keys_for_pool)

            # Saving logic remains the same
            filename = folder_path / \
                f"LRSSM_simulation_{t}_{m[0]}_{int(l*100)}.pkl"

            with open(filename, 'wb') as f:
                pickle.dump(results_list, f)

            # delete over memory
            del results_list
            gc.collect()

            print('...done')

# %% [RUN MAIN] run: $python LRSSM_simulations.py


if __name__ == '__main__':

    # Output folder exist
    folder_path = Path(f"./output/simulation")

    # check that the folder exixst
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print("New folder created successfully!")

    main(folder_path)
