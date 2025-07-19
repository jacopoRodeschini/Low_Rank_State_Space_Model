#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 30, 2025

@author: jacopo
@title : LSSM simulation v4 (fix those border effects by enlarging the latent dimension during the M-step)
"""

import filter_jax as filterjax
from spatial import spdeAppoxCov
import os
import sys
import pickle
import gc
from itertools import product

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from joblib import Parallel, delayed

# Remove unused imports:
# import traceback, subprocess, pandas as pd, matplotlib, mpl_toolkits, etc.
# Only keep what's necessary in this script.

# Update JAX config
jax.config.update("jax_enable_x64", False)

# Environment-specific paths
if os.cpu_count() == 8:
    geopydir = '/home/jacopo/Documents/Dottorato/geopy/geopy'
    filterdir = '/home/jacopo/Documents/Dottorato/geopy/geopy/filter_py/'
else:
    geopydir = '/home/jrodeschini/geopy'
    filterdir = '/home/jrodeschini/geopy/filter_py'

sys.path.insert(0, geopydir)
sys.path.insert(0, filterdir)


# Assuming task is defined at bottom of file or in filter_jax


class KeyStream:
    def __init__(self, seed):
        self._key = seed

    def next(self, num=None):
        if num is None:
            new_key, self._key = random.split(self._key)
            return new_key
        else:
            keys = random.split(self._key, num=num + 1)
            self._key = keys[-1]
            return keys[:-1]


def run_simulation():
    nvar = 3
    nlat = 2
    domain = np.array([1, 1, 1])
    beta = np.array([1, 2, -1])
    s2 = 1
    ks = np.sqrt(8) * np.array([7, 2])
    nu = np.ones((nlat, 1))
    A = np.array([[0.5, 1], [0.5, 0.3], [0.2, 0.8]])
    flatent = [0.8, -0.5]
    s2error = np.array([0.5, 1.5, 1])

    T = [25, 50]
    mtrain = [[10, 10, 10], [15, 15, 15], [20, 20, 20]]
    mtotal = [25, 25, 25]

    lowrank = [1, 0.75, 0.5, 0.25, 0.15]
    boot = 100

    param = sorted(list(product(mtrain, T)), key=lambda x: x[1])

    main_key = random.PRNGKey(1234)
    sub_key = random.split(main_key, num=len(param))

    for inx, (m, t) in enumerate(param):
        loop_key, main_key = random.split(sub_key[inx, :])

        for l in lowrank:
            loop_key, main_key = random.split(main_key)
            keys_for_pool = random.split(loop_key, num=boot)

            print(f"Processing combination m={m}, t={t}, l={l}")

            results_list = Parallel(n_jobs=25, backend='loky')(
                delayed(task)(key, domain=domain, mtotal=mtotal, mtrain=m, A=A, ks=ks, beta=beta, T=t,
                              s2error=s2error, flatent=flatent, lowrank=l, max_iter=300,
                              tol_relat=1e-6, verbose=False, same=True, regular=True, estimates=False)
                for key in keys_for_pool)

            filename = f'/home/jrodeschini/geopy_casestudy/geopy_grins/output/JAX_LSSMsim_v4_{t}_{m[0]}_{int(l*100)}_v3.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(results_list, f)

            del results_list
            gc.collect()
            print('...done')


if __name__ == "__main__":
    run_simulation()
