#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:21:26 2025

@author: jacopo
@
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# %%

# paths = [p.name for p in Path(".").glob("JAX*") if p.suffix in {".pkl", ".pickle"} ]

lowrank = 0.75  # 1, 0.75, 0.5, 0.25, 0.15

for lowrank in [1, 0.75, 0.5, 0.25, 0.15]:
    file_paths = [p.absolute() for p in Path(
        "/home/jrodeschini/geopy_casestudy/geopy_grins/output").glob(f'JAX*_v{4}_*_{int(lowrank*100)}_v3.pkl')]

    # paths = ['LSSMsim_v4_100_400_fixk.pkl']
    data = {}
    for path in file_paths:

        with open(path, "rb") as f:  # "rb" = read-binary
            results = pickle.load(f)

        # filter the none
        results = [r for r in results if r is not None]
        results = [r for r in results if r['niter'] < 299]

        results = results[:100]

        # True parameter values
        true_values = {
            'beta': np.array([1.0, 2, -1.0]),
            's2e': np.array([0.5, 1.5, 1]),
            'ks': np.array([7 * np.sqrt(8), 2 * np.sqrt(8)]),
            # flattened 2x2 matrix
            'est_A': np.array([0.5, 1., 0.5, 0.3, 0.2, 0.8]),
            'f': np.array([0.85, -0.5])
        }

        # Accumulate estimates
        params = {
            'beta': [],
            's2e': [],
            'ks': [],
            'est_A': [],
            'f': []
        }

        for i, r in enumerate(results):
            f_est = np.array(r['f'])
            est_A = np.array(r['est_A']).reshape(3, 2)
            # Check if f is not in decreasing order
            if not np.all(np.diff(f_est) <= 0):
                order = np.argsort(-f_est)  # Sort indices to make f decreasing
                # Reorder and update the values in-place
                results[i]['f'] = f_est[order]
                # Ensure W is 2D: shape (obs_dim, num_components)
                est_A = est_A[:, order]  # reorder columns
                # Reorder ks accordingly
                ks = np.array(r['ks'])
                results[i]['ks'] = ks[order]
            # Flip sign of columns where the first row is negative
            for j in range(est_A.shape[1]):
                if est_A[0, j] < 0:
                    est_A[:, j] *= -1

            # store as flattened array again
            results[i]['est_A'] = est_A.flatten()

        # print([r['niter'] for r in results])
        # results = [r for r in results if r['deltaL']<0.1]
        # print(len(results))
        for r in results:
            if r['f'][1] < 0:
                params['beta'].append(r['beta'])
                params['s2e'].append(r['s2e'])
                params['ks'].append(r['ks'])
                params['est_A'].append(r['est_A'])
                params['f'].append(r['f'])

        # Convert to numpy arrays
        for key in params:
            params[key] = np.array(params[key])

        def compute_bias(estimates, true_val):
            return np.mean(estimates - true_val, axis=0)

        def compute_elementwise_rmse(estimates, true_val):
            mse = np.mean((estimates - true_val) ** 2, axis=0)
            rmse = np.sqrt(mse)
            return rmse / np.abs(true_val)

        results_summary = {}

        for key in params:
            estimates = params[key]
            true_val = true_values[key]

            # Ensure true_val is a numpy array for consistency
            true_val_array = np.array(true_val)

            # Compute bias element-wise
            bias = compute_bias(estimates, true_val_array)

            # Scalar vs vector handling
            if np.isscalar(true_val) or true_val_array.ndim == 0:
                rmse = np.sqrt(np.mean((estimates - true_val) ** 2))
                norm_rmse = rmse / np.abs(true_val)
            else:
                # Vector RMSE: sqrt(mean(||estimate - true||^2)) across samples
                diffs = estimates - true_val_array  # shape (N, D)
                l2_squared = np.sum(diffs ** 2, axis=1)  # shape (N,)
                rmse = np.sqrt(np.mean(l2_squared))
                norm_rmse = rmse / np.linalg.norm(true_val_array)

            results_summary[key] = {
                'bias': bias,
                'normalized_RMSE': norm_rmse
            }

        t = results[0]['T_lenght']
        # m = int(results[0]['m'][0])
        m = int(path.stem.split('_')[4])
        # print(t, m)

        rmse_train = [r['rmse_train'] for r in results]
        rmse_test = [r['rmse_test'] for r in results]
        time_tot = [r['time_tot'] for r in results]

        data[(f"T={t}", f"m={m}", "Bias")] = {
            "beta_1": results_summary['beta']['bias'][0],
            "beta_2": results_summary['beta']['bias'][1],
            "beta_3": results_summary['beta']['bias'][2],
            'sigma_1^2': results_summary['s2e']['bias'][0],
            'sigma_2^2': results_summary['s2e']['bias'][1],
            'sigma_3^2': results_summary['s2e']['bias'][2],
            'k_1': results_summary['ks']['bias'][0],
            'k_2': results_summary['ks']['bias'][1],
            'W_1': results_summary['est_A']['bias'][0],
            'W_2': results_summary['est_A']['bias'][1],
            'W_3': results_summary['est_A']['bias'][2],
            'W_4': results_summary['est_A']['bias'][3],
            'W_5': results_summary['est_A']['bias'][4],
            'W_6': results_summary['est_A']['bias'][5],
            'f_1': results_summary['f']['bias'][0],
            'f_2': results_summary['f']['bias'][1],
            'rmse_train': np.mean(rmse_train),
            'rmse_test': np.mean(rmse_test),
            'time': np.mean(time_tot)
        }

        data[(f"T={t}", f"m={m}", "RMSE")] = {
            "beta_1": results_summary['beta']['normalized_RMSE'],
            "beta_2": None,
            "beta_3": None,
            'sigma_1^2': results_summary['s2e']['normalized_RMSE'],
            'sigma_2^2': None,
            'sigma_3^2': None,
            'k_1': results_summary['ks']['normalized_RMSE'],
            'k_2': None,
            'W_1': results_summary['est_A']['normalized_RMSE'],
            'W_2': None,
            'W_3': None,
            'W_4': None,
            'W_5': None,
            'W_6': None,
            'f_1': results_summary['f']['normalized_RMSE'],
            'f_2': None,
            'rmse_train': np.std(rmse_train),
            'rmse_test': np.std(rmse_test),
            'time': np.std(time_tot)
        }

        # print(path,len(data))
    esti_table = pd.DataFrame(data)

    index_sort = [
        (f'T={T}', f'm={m}', metric)
        for T in [25, 50]
        for m in [10, 15, 20]
        for metric in ['Bias', 'RMSE']
    ]

   #  index_sort = index_sort[:-2]

    esti_table = esti_table[index_sort]

    # svate the csv
    filepath = f"/home/jrodeschini/geopy_casestudy/geopy_grins/output/Simulation_table_R{int(lowrank*100)}_v3.pkl"
    # esti_table.to_csv(filepath, index=False, encoding='utf-8', sep=';')
    with open(filepath, 'wb') as f:
        pd.to_pickle(esti_table, filepath)
        # pickle.dump(esti_table, f)

    print(lowrank, "...done")


# %% Import the dataframe (local)
lowrank = 1
filepath = f"/home/jacopo/Documents/Dottorato/geopy_casestudy/geopy_grins/output/Simulation_table_R{int(lowrank*100)}_v3.pkl"

with open(filepath, 'rb') as f:
    esti_table = pd.read_pickle(f)
