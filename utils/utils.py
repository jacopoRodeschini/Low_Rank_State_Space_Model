#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 30, 2025

@author: jacopo
"""

import sys
import os
import pickle
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Point, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import pandas as pd
import time
from scipy.optimize import minimize
# from jax.scipy.optimize import minimize
from gstools.covmodel import Matern
from jax.scipy.linalg import block_diag
from scipy.sparse import linalg as sp_linalg
import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial


import jax
jax.config.update("jax_enable_x64", False)

# import gstools as gs


# %% key stream


class KeyStream:
    """A simple helper class to manage JAX random keys."""

    def __init__(self, seed):
        self._key = seed

    def next(self, num=None):
        """Gets a new key, updating the internal state."""
        # Unce use the key is update

        if num is None:
            new_key, self._key = jax.random.split(self._key)
        else:
            keys = jax.random.split(self._key, num=num+1)
            new_key = keys[:-1]
            self._key = keys[-1]

        return new_key

# %% joblib task [estimation]


def task(seed, domain, mtotal, mtrain, A, ks, beta, T, s2error, flatent, max_iter=100, tol_relat=1e-6, same=True,
         regular=True, lowrank=1, angle_thr=2, verbose=True, estimates=False):

    try:

        target_dtype = jnp.float32

        domain = jnp.asarray(domain, dtype=target_dtype)
        A = jnp.asarray(A, dtype=target_dtype)
        ks = jnp.asarray(ks, dtype=target_dtype)
        beta = jnp.asarray(beta, dtype=target_dtype)
        s2error = jnp.asarray(s2error, dtype=target_dtype)
        flatent = jnp.asarray(flatent, dtype=target_dtype)

        mtotal = jnp.asarray(mtotal, dtype=jnp.int32)
        mtrain = jnp.asarray(mtrain, dtype=jnp.int32)

        # Initalisate the keys stream generator
        keys = KeyStream(seed)

        # ---- 1: Create a random dataset (HDGM)
        nvar = len(s2error)
        nlat = len(flatent)

        y_true, x_true, Xbeta, points, index_train, block_p, pdim = buildDataset(
            keys, domain, mtotal, mtrain, A, ks, beta, T, s2error, flatent, regular=True)

        # Training points
        m_points = [pt[inx, :] for pt, inx in zip(points, index_train)]

        # Dimension of the latent equation
        max_vertices = [int(np.ceil(lowrank * len(pt)))
                        for pt in m_points]  # observed

        # Boundary polygon
        hull = [ConvexHull(pt) for pt in m_points]
        boundary_polygon = [Polygon(h.points[h.vertices])
                            for h in hull]  # .buffer(0.01)
        boundary_vertices = [h.vertices for h in hull]

        # Compute the outer points
        delta = 0.5
        x = [jnp.linspace(start=h.min_bound[0]-delta,
                          stop=h.max_bound[0]+delta, num=15) for h in hull]
        y = [jnp.linspace(start=h.min_bound[1]-delta,
                          stop=h.max_bound[1]+delta, num=15) for h in hull]

        outer_grid = [jnp.meshgrid(xi, yi) for xi, yi in zip(x, y)]

        outer_points = [jnp.vstack(
            (gr[0].flatten(), gr[1].flatten())).T for gr in outer_grid]  # Mesh vertex grid

        # remove the outer point within the polygon
        index = [bd_poly.contains([Point(p) for p in ot_points])
                 for bd_poly, ot_points in zip(boundary_polygon, outer_points)]
        outer_points = [ou_points[~inx]
                        for ou_points, inx in zip(outer_points, index)]

        # compute the all_points (the inner points index and the boundary coordinates)
        all_points = [jnp.vstack((vertex, ou_points))
                      for vertex, ou_points in zip(m_points, outer_points)]

        # Reduce the vertex count and compute a valid delanuay
        delaunay = [reduce_vertex_count_hard(
            all_pt, bd_poly, m_v) for all_pt, bd_poly, m_v in zip(all_points, boundary_polygon, max_vertices)]

        min_angle = [np.round(np.min([np.min(compute_angles(d.points, t))
                                     for t in d.simplices]), 2) for d in delaunay]

        # low-rank fcov
        sm_points = []  # smoothed points
        fcov = []      # Covariance functions
        for i in range(len(delaunay)-1):

            if lowrank < 1:
                new_points, min_angle = laplacian_smoothing(
                    delaunay[i], boundary_polygon[i], max_iterations=10, angle_thr=30, verbose=verbose)
            else:
                new_points = delaunay[i].points

            # build the covariance mesh
            inner = isinner(boundary_polygon[i], new_points)

            temp = spdeAppoxCov(new_points, delaunay[i].simplices, outer_index=~inner, add_boundary=True,
                                latlon=False, uniformRef=False, rescale=10)

            fcov.append(temp)

        # get the test index
        index_test = [np.arange(pdim[i])[~np.isin(
            np.arange(pdim[i]), index_train[i])] for i in range(nvar)]

        y_train = np.vstack(
            [y_true[block_p[i]:block_p[i+1], :][index_train[i], :] for i in range(nvar)])
        y_test = np.vstack([y_true[block_p[i]:block_p[i+1], :]
                           [index_test[i], :] for i in range(nvar)])
        points_test = [points[i][index_test[i], :] for i in range(nvar)]

        Xbeta_train = np.vstack(
            [Xbeta[block_p[i]:block_p[i+1], :, :][index_train[i], :, :] for i in range(nvar)])
        Xbeta_test = np.vstack(
            [Xbeta[block_p[i]:block_p[i+1], :, :][index_test[i], :, :] for i in range(nvar)])

        pdim_train = jnp.array([len(inx) for inx in index_train])
        block_p_train = np.hstack((0, np.cumsum(pdim_train)))
        qdim_train = jnp.array([cov.n_inner_points for cov in fcov], jnp.int32)
        block_q_train = jnp.hstack((0, jnp.cumsum(qdim_train)))

        # ---- 4: Estimate
        beta0 = jax.random.uniform(keys.next(), shape=beta.shape)
        s2e0 = jax.random.uniform(
            keys.next(), shape=s2error.shape, minval=0.1, maxval=2)
        f0 = jnp.sort(jax.random.uniform(
            keys.next(), shape=flatent.shape, minval=0.2, maxval=0.8))[::-1]
        f0 = f0.at[1].set(-f0[1])

        A0 = jax.random.uniform(
            keys.next(), shape=A.shape, minval=0.2, maxval=2)
        ks0 = jnp.sort(jnp.sqrt(8)*jax.random.uniform(keys.next(),
                       shape=ks.shape, minval=1, maxval=7))[::-1]
        x0 = jax.random.uniform(keys.next(), minval=0,
                                maxval=1, shape=(block_q_train[-1],))

        est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_covList, est_A, nstat, y_hat, x_T, P_T = fit(
            y_train, fcov, block_p_train, pdim_train, Xbeta_train, m_points, max_iter=max_iter, tol_par=1e-3, tol_relat=tol_relat, nstat=[],
            beta0=beta0, s2e0=s2e0, f0=f0, A0=A0, ks0=ks, x0=x0, Sigma0=None, verbose=verbose)

        # compute the train RMSE
        rmse_train = jnp.sqrt(jnp.mean((y_train - y_hat).flatten()**2))

        # Compute the full RMSE
        basis = buildBasis_list(points_test, est_covList)

        H = buildH_dense(est_A, basis)
        y_hat_test = predict(H, x_T, Xbeta=Xbeta_test, beta=est_beta)
        rmse_test = jnp.sqrt(jnp.mean((y_test - y_hat_test).flatten()**2))

        # total time: remove the first iteration (compiling time)
        time_tot = sum(
            [it['time_tot'] if 'time_tot' in it else 0 for it in nstat[2:]])
        time_tot += np.mean([it['time_tot']
                            if 'time_tot' in it else 0 for it in nstat[2:]])

        niter = nstat[-1]['niter']
        res = nstat[-1]
        res['seed'] = seed
        res['lowrank'] = lowrank
        res['m'] = mtrain
        res['T_lenght'] = T
        res['domain'] = domain
        res['regolar'] = regular
        res['time_tot'] = time_tot
        res['rmse_train'] = rmse_train
        res['rmse_test'] = rmse_test

        gc.collect()  # Help free memory
        if estimates:
            return res, points, pdim, qdim, y_train, Xbeta_train, est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_covList, est_A, nstat, y_hat, x_T, P_T
        else:
            return res

    except Exception as e:
        print(e)
        gc.collect()  # Help free memory
        # If anything crashes, this block will catch it.
        return None

# %% Build dataset


def predict(H, x_T, P_T=None, Xbeta=None, beta=None):
    p, q = H.shape
    T = x_T.shape[1]-1

    # convert all at float32
    H = np.asarray(H, dtype=np.float32)
    x_T = np.asarray(x_T, dtype=np.float32)

    if Xbeta is None:
        Xbeta = np.zeros((p, 1, T))
        beta = np.ones((1, ))

    y_hat = np.zeros((p, T), dtype=np.float32)
    for t in range(T):
        y_hat[:, t] = Xbeta[:, :, t] @ beta + H @ x_T[:, t+1]

    if P_T is not None:
        Sigma_y_hat = np.zeros((p, p, T), dtype=np.float32)
        for t in range(T):
            Sigma_y_hat[:, :, t] = H @ P_T[:, :, t+1] @ H.T
        return y_hat, Sigma_y_hat

    else:
        return y_hat


def buildDataset(keys, domain, mtotal, mtrain, A, ks, beta, T, s2error, flatent, regular=True):

    if any(mtotal < mtrain):
        print("m total should be >= m train")

    # simulate the dataset (m, m*)
    nvar = len(domain)
    nlat = len(ks)

    # m total random points
    points = []

    for i in range(nvar):
        if regular:
            v = jnp.linspace(
                start=0, stop=domain[i], num=mtotal[i], dtype=jnp.float32)
            gridx, gridy = jnp.meshgrid(v, v)
            points.append(jnp.vstack(
                (gridx.flatten(), gridy.flatten())).T)  # Mesh vertex grid
        else:

            points = jnp.random.uniform(
                0, domain[0], (mtotal[0]**2, 2), key=keys.next(), dtype=jnp.float32)

    # Get the (random) traininig index
    index_train = []
    for i in range(nvar):
        index_train.append(jax.random.choice(keys.next(), np.arange(
            len(points[i])), shape=(mtrain[i]**2,), replace=False))

    # Build the structure matrix
    pdim = jnp.array([len(pt) for pt in points])
    qdim = jnp.array([len(points[i]) for i in range(len(flatent))])
    block_p = jnp.hstack((0, jnp.cumsum(pdim)))

    # use the True Matern (to simulate the HDGM) [take care about the dimensions!]
    fcov = [Matern(dim=2, var=1, len_scale=1, rescale=ksi,  nu=1) for ksi in ks]

    # the mesh is just useful to get the basis matrix H = I_n
    # the simulation will work over the (m, m*) dataset = points
    # all_points = np.unique(np.vstack(points), axis=0)

    # plot
    # fig, ax = plt.subplots()
    # ax.plot(points[0][:, 0], points[0][:, 1], 'x')
    # ax.plot(points[0][index_train[0], 0], points[0][index_train[0], 1], 'o')

    mesh = [spdeAppoxCov(pt, latlon=False, uniformRef=False,
                         add_boundary=False, rescale=ksi) for pt, ksi in zip(points, ks)]

    # fig, ax = plt.subplots()
    # mesh[0].plot_mesh(ax=ax)
    basis = buildBasis_list(points, mesh)
    H = buildH_dense(A, basis)

    # Compute the Matern covariane
    Q = block_diag(*[f.covariance(mh.get_distance())
                   for f, mh in zip(fcov, mesh)])

    Xbeta = block_diag_3D(
        [jax.random.normal(keys.next(), shape=(pi, 1, T), dtype=jnp.float32) for pi in pdim])

    # ---- Build the R, F matrix [including alpha]
    R, F = buildRF_dense(s2error, flatent, pdim, qdim)

    # define initial values of beta
    x0 = jnp.repeat(jnp.ones(nlat), qdim)
    Sigma0 = block_diag(*[jnp.eye(q) for q in qdim])

    # simulate the dataset
    y_true, x_true = sim(keys, R, F, H, Q, x0, Sigma0, Xbeta, beta)

    gc.collect()
    return y_true, x_true, Xbeta, points, index_train, block_p, pdim


def sim(keys, R, F, H, Q, x0, Sigma0, Xbeta, beta):
    """
    Simulates a time series from the state-space model using JAX and a Python for-loop.
    This version does NOT use JIT compilation and is therefore slower.

    Args:
        key: is a JAX PRNGKey strem object (next methods).
        ... other model parameters.

    Returns:
        y_t : (p, T) JAX array of simulated observations
        x_t : (q, T+1) JAX array of simulated state vectors [x_0, ..., x_T]
    """
    # Get dimensions from input shapes
    T = Xbeta.shape[2]
    p = R.shape[0]
    q = F.shape[0]

    # Pre-compute Cholesky decompositions
    chol_R = jnp.linalg.cholesky(R)
    chol_Q = jnp.linalg.cholesky(Q)
    chol_Sigma0 = jnp.linalg.cholesky(Sigma0)

    # --- Initial State (t=0) ---
    initial_noise = jax.random.normal(keys.next(), shape=(q,))
    x_current = x0 + chol_Sigma0 @ initial_noise  # This is state x_0

    # --- Simulation using a Python for-loop ---
    # Since JAX arrays are immutable, we build Python lists of the results
    # and stack them into a single array at the end.
    x_history = [x_current]
    y_history = []

    # Loop T times to generate T observations (y_0, ..., y_{T-1})
    for t in range(T):
        # 1. Generate the observation y_t based on the current state x_t
        # Note: The original code had a slight lookahead (y_{t-1} from x_t).
        # This version uses the more standard y_t from x_t.
        obs_noise = chol_R @ jax.random.normal(keys.next(), shape=(p,))
        mean_reg = Xbeta[:, :, t] @ beta
        y_t = mean_reg + H @ x_current + obs_noise
        y_history.append(y_t)

        # 2. Evolve the state to the next step: x_{t+1} from x_t
        process_noise = chol_Q @ jax.random.normal(keys.next(), shape=(q,))
        x_next = F @ x_current + process_noise

        # 3. Store the new state and update the current state for the next loop iteration
        x_history.append(x_next)
        x_current = x_next

    # --- Final Assembly ---
    # Convert the lists of arrays into single JAX arrays with the correct shape.
    # jnp.stack(..., axis=1) is equivalent to np.array(...).T
    final_y_t = jnp.stack(y_history, axis=1)
    final_x_t = jnp.stack(x_history, axis=1)

    gc.collect()
    return final_y_t, final_x_t


@jax.jit
def block_diag_3D(blocks):
    """
    Creates a 3D block diagonal matrix from a list of 3D JAX arrays.

    This function is JIT-compatible. Each input array must be of shape
    (n_i, m_i, p), where the last dimension `p` is constant.

    Args:
        blocks: A list of 3D JAX arrays.

    Returns:
        A single 3D JAX array with the input blocks arranged on the diagonal.
    """
    # Guard against an empty list of blocks
    if not blocks:
        # Return an empty array. The shape is ambiguous, so a 0-element
        # array is the most sensible default.
        return jnp.array([])

    # Determine the total shape for the output matrix
    total_shape_0 = sum(arr.shape[0] for arr in blocks)
    total_shape_1 = sum(arr.shape[1] for arr in blocks)

    # The last dimension is constant, take it from the first block
    p = blocks[0].shape[2]

    # Use jnp.result_type to handle different input dtypes (e.g., int, float32)
    dtype = jnp.result_type(*blocks)

    # Initialize the block diagonal matrix with zeros
    out = jnp.zeros((total_shape_0, total_shape_1, p), dtype=dtype)

    # Current start index for the first two dimensions
    curr_idx_0 = 0
    curr_idx_1 = 0

    for arr in blocks:
        n_i, m_i, _ = arr.shape

        # This is the key JAX-native update.
        # It creates a *new* array `out` by updating the specified slice.
        # The syntax is designed to look like NumPy but is functionally pure.
        out = out.at[curr_idx_0: curr_idx_0 + n_i,
                     curr_idx_1: curr_idx_1 + m_i,
                     :].set(arr)

        curr_idx_0 += n_i
        curr_idx_1 += m_i

    return out

# %% fit


def fit(y_t, est_covList, block_p, pdim, Xbeta, points, beta0=None, s2e0=None,
        f0=None, A0=None, ks0=None, x0=None, Sigma0=None,
        max_iter=100, tol_par=1e-2, tol_relat=1e-3, nstat=[], verbose=True):

    # make sure all think are jax.numpy
    # y_t = jnp.asarray(y_t, dtype=jnp.float32)
    # Xbeta = jnp.asarray(Xbeta, dtype=jnp.float32)
    # pdim = jnp.asarray(pdim)

    # get global constant
    nvar = len(pdim)
    nlat = len(est_covList)

    # create the vector_parameter
    est_ks = jnp.array([cov.rescale for cov in est_covList])
    # stiff = [f.stiff.toarray() for f in est_covList]
    # mass = [f.mass.toarray() for f in est_covList]

    # da costruire usando la discretizzazione della latente [with no boundary]
    qdim = jnp.array(
        [cov.n_inner_points for cov in est_covList], dtype=jnp.int32)
    block_q = jnp.hstack((0, jnp.cumsum(qdim)))

    p, T = y_t.shape
    q = jnp.sum(qdim)

    # set the initial values
    est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_A = getInitialValues(
        y_t, Xbeta, tuple(block_p.tolist()), tuple(block_q.tolist()), T)

    fix_b = False
    fix_s2 = False
    fix_f = False
    fix_A = False
    fix_ks = False
    fix_x0 = False
    fix_Sigma0 = False

    if beta0 is not None:
        est_beta = jnp.asarray(beta0)
        # fix_b = True
    if s2e0 is not None:
        est_s2e = jnp.asarray(s2e0)
        # fix_s2 = True
    if f0 is not None:
        est_f = jnp.asarray(f0)
        # fix_f = True
    if A0 is not None:
        est_A = jnp.asarray(A0)
        # fix_A = True
    if ks0 is not None:
        est_ks = jnp.asarray(ks0)
        for fcov, ksi in zip(est_covList, est_ks):
            fcov.rescale = ksi

        # fix_ks = True
    if x0 is not None:
        est_x0 = jnp.asarray(x0)
        # fix_x0 = True
    if Sigma0 is not None:
        est_Sigma0 = jnp.asarray(Sigma0)
        # fix_Sigma0 = True
    # ---- print messages
    if verbose:
        msg = f"beta:{jnp.round(est_beta,2)} - s2e:{jnp.round(est_s2e,2)} - f:{jnp.round(est_f,2)} - rescale:{jnp.round(est_ks,2)} - A: {jnp.round(est_A.flatten(),2)}"
        print(msg)

    est_vet_par = jnp.hstack(
        (est_beta.flatten(), est_s2e.flatten(), est_f.flatten(), est_ks.flatten(), est_A.flatten()))

    # Flag of the EM convergence
    flag = True
    logL_prev = -jnp.inf
    niter = 0

    it = {'niter': niter, 'beta': est_beta, 's2e': est_s2e, 'f': est_f,
          'ks': est_ks, 'est_A': est_A.flatten(), 'x0': est_x0.mean(),
          'S0': jnp.diag(est_Sigma0).mean(), 'logL': logL_prev, 'deltaP': np.inf,
          'deltaL': np.inf}
    nstat.append(it)

    # Compute the basis matrix (just one) - no boundary
    basis = buildBasis_list(points, est_covList)
    Phi = buildH_dense(jnp.ones((nvar, nlat), dtype=jnp.float32), basis)

    # stiff, mass

    # Start EM iteration
    while flag:
        niter += 1

        # ---- build parametrised matrices
        tStart_iter = time.time()
        H = buildH_dense(est_A, basis)  # dense

        # R, F = buildRF(est_s2e, est_f, pdim, qdim)
        R, F = buildRF_dense(est_s2e, est_f, pdim, qdim)

        # Depend by the model (precision matrix) [Sigma = dense matrix]
        # 1) the ks is update by the previous itration
        # 2) the Q is built considering the boudary [and cut later on the inner points only]

        invQ = [jnp.asarray(fcov.precision()[:fcov.n_inner_points, :fcov.n_inner_points].toarray(), dtype=jnp.float32)
                for fcov in est_covList]

        Q = block_diag(
            *[jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=jnp.float32)) for mt in invQ])
        # Q = compute_Q_jax(est_ks, stiff, mass, tuple(ninner.tolist()))

        # ---- E step
        y_hat, x_t, x_T, P_T, P_T_1, S11, S10, S00, logL_cur, tdelta_Edet = E_step(
            y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta)

        # print("E-step f", np.round(np.sum((x_true[1:,] - x_t[1:,])**2), 2))
        # print("E-step s", np.round(np.sum((x_true[1:,] - x_T[1:,])**2), 2))

        ##

        # ---- M step
        # you need to use the est_covList with boundary -> to update ks
        update_beta, update_s2e, update_f, update_x0, update_Sigma0, est_covList, update_A, tdelta_Mdet = M_step(
            y_t, y_hat, F, H, Xbeta, points, est_covList, block_p, block_q, x_T, P_T, S11, S10, S00, Phi, est_beta)

        if fix_b:
            update_beta = est_beta
        if fix_s2:
            update_s2e = est_s2e
        if fix_f:
            update_f = est_f
        if fix_A:
            update_A = est_A
        if fix_x0:
            update_x0 = est_x0
        if fix_Sigma0:
            update_Sigma0 = est_Sigma0
        if fix_ks:
            for f, ksi in zip(est_covList, est_ks):
                f.rescale = ksi

        update_ks = jnp.array(
            [cov.rescale for cov in est_covList], dtype=jnp.float32)

        # Stack the vector_parameter
        update_vet_par = np.hstack(
            (update_beta.flatten(), update_s2e.flatten(), update_f.flatten(), update_ks.flatten(), update_A.flatten()))

        # compute the delta log_likelihood ( 0 < current - previous < tol_li )
        # and check the stop criterio
        delta_par = np.linalg.norm(est_vet_par - update_vet_par, 2)
        delta_lik = logL_cur - logL_prev

        relat_lik = jnp.abs(delta_lik / logL_prev)

        # ---- print messages
        tdelta_iter = time.time() - tStart_iter

        if verbose:
            temp = f'''iter:{niter}, logl: {jnp.round(logL_cur,2)}, deltL: {jnp.round(delta_lik,2)}, relatL: {jnp.round(relat_lik,5)}
beta:{np.round(update_beta,2)},s2e:{jnp.round(update_s2e,2)}, f:{jnp.round(update_f,2)},rescale:{jnp.round(update_ks,2)},
A:{est_A.flatten()}, x0={jnp.round(est_x0.mean(),2)}, S0 = {np.round(jnp.diag(est_Sigma0).mean(),2)}'''
            print(temp)
            print(tdelta_iter, tdelta_Edet.sum(), tdelta_Mdet.sum())

        if niter == max_iter or relat_lik <= tol_relat:  # or delta_par <= tol_par
            flag = False
        else:
            # Update the paramiters for the next iterations
            est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A = update_beta, update_s2e, update_f, update_x0, update_Sigma0, update_ks, update_A
            est_vet_par = update_vet_par
            logL_prev = logL_cur

            # append the results
            it = {'niter': niter, 'beta': est_beta, 's2e': est_s2e, 'f': est_f,
                  'ks': est_ks, 'est_A': est_A.flatten(), 'x0': est_x0.mean(),
                  'S0': jnp.diag(est_Sigma0).mean(), 'logL': logL_cur,
                  'deltaP': delta_par, 'deltaL': delta_lik, 'relatL': relat_lik,
                  'time_tot': tdelta_iter, 'tdelta_E': tdelta_Edet.sum(),
                  'tdelta_E_detail': tdelta_Edet, 'tdelta_M': tdelta_Mdet.sum(), 'tdelta_M_detail': tdelta_Mdet}

            nstat.append(it)

    gc.collect()
    return est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_covList, est_A, nstat, y_hat, x_T, P_T

# %% E step


def E_step(y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta):

    # python
    tStart = time.time()
    x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL = filterjax.filter_smw_nan(
        y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta)
    jax.block_until_ready(x_t)
    tdelta_kf = time.time() - tStart

    tStart = time.time()
    x_T, P_T, P_T_1 = filterjax.smoother_smw(
        H, F, x_t, P_t, K, x_t_1, P_t_1, invP_t_1)
    jax.block_until_ready(x_T)
    tdelta_sm = time.time() - tStart

    # Output Eq: (7a, 7b, 7c, 7d, 7e)
    tStart = time.time()
    y_hat, S11, S10, S00 = filterjax.computeExpectedValues(
        y_t, x_T, P_T, P_T_1, R, H, Xbeta, est_beta)
    jax.block_until_ready(y_hat)
    tdelta_exp = time.time() - tStart

    tdelta = np.array([tdelta_kf, tdelta_sm, tdelta_exp], dtype=jnp.float32)
    return y_hat, x_t, x_T, P_T, P_T_1, S11, S10, S00, logL, tdelta

# %% M step


def M_step(y_t, y_hat, F, H, Xbeta, points, est_covList, block_p, block_q, x_T,
           P_T, S11, S10, S00, Phi, est_beta):

    # convert all input to save memory
    p, T = y_t.shape
    q = block_q[-1]
    b = Xbeta.shape[1]
    nvar = len(block_p)-1
    nlat = len(block_q)-1
    ndim = block_p[1:] - block_p[:-1]  # observed dimensions
    ldim = block_q[1:] - block_q[:-1]  # latent dimensions

    # F = sp.diags(Fdiag, 0, format='csr', dtype=np.float32)
    # Update the parametes first the the F in order to do the flip

    # Update f (Eq 3a)
    est_f = jnp.zeros((nlat))
    tStart = time.time()
    for q in range(nlat):

        s = slice(block_q[q], block_q[q+1])
        # num = np.trace(S10[block_q[q]:block_q[q+1], block_q[q]:block_q[q+1]])
        # den = np.trace(S00[block_q[q]:block_q[q+1], block_q[q]:block_q[q+1]])
        num = jnp.trace(S10[s, s])
        den = jnp.trace(S00[s, s])

        # print(q, num, den)
        est_f = est_f.at[q].set(num / den)
    tdelta_f = time.time() - tStart

    # beta (the same as Eq.5 Calculli)
    err = y_t - y_hat  # compute the prediction error

    # Update beta (OLS)
    # cond_xy = np.einsum('mbt, mt-> b', self.X, y)
    # Update beta (OLS)
    # tStart = time.time()
    # beta = compute_beta(b, y_t, x_T, H, T, Xbeta)
    # tdelta_beta = time.time() - tStart

    tStart = time.time()
    beta = compute_beta_jax(b, y_t, x_T, H, Xbeta)
    jax.block_until_ready(beta)
    tdelta_beta = time.time() - tStart

    # s2 (see Eq, 14c)
    # compute the s2e
    # tStart = time.time()
    # s2e = compute_s2e(
    #     err, H, P_T, block_p, nvar, T, ndim)
    # tdelta_s2e = time.time() - tStart

    tStart = time.time()
    s2e = compute_s2e_jax(err, H, P_T, tuple(block_p.tolist()))
    jax.block_until_ready(s2e)
    tdelta_s2e = time.time() - tStart

    # update A
    # est_A, tdelta_A = compute_A2(
    #     y_t, Xbeta, est_beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi)

    tStart = time.time()

    est_A = compute_A2_jax(y_t, Xbeta, beta, x_T, P_T,
                           tuple(block_p.tolist()), tuple(block_q.tolist()),  nvar, nlat, T, ldim, Phi)
    # est_A = np.array(est_A.copy())

    jax.block_until_ready(est_A)
    tdelta_A = time.time() - tStart

    # Set the parameter of the minimise object

    tStart = time.time()
    Omega = S11 - S10 @ F.T - F @ S10.T + F @ S00 @ F.T
    par0 = jnp.log(
        jnp.array([fcov.rescale for fcov in est_covList], dtype=jnp.float32))

    opt = minimize(minf, par0, args=(est_covList, T, Omega), method='BFGS',  # method='L-BFGS-B'
                   tol=1e-3, jac=False, options={'maxiter': 100})

    tdelta_ks = time.time() - tStart

    # #update the covraiance object
    # for i in range(len(est_covList)):
    #     est_covList[i].rescale = ks[i]

    # est_covList[0].rescale = 0.17
    # update the initial state mu and variance
    x0 = x_T[:, 0]
    Sigma0 = P_T[:, :, 0]

    tdelta = jnp.array([tdelta_beta, tdelta_s2e, tdelta_f,
                       tdelta_ks, tdelta_A], dtype=jnp.float32)
    return beta, s2e, est_f, x0, Sigma0, est_covList, est_A, tdelta

# %% [JAX] updating formula


@partial(jit, static_argnames=['b'])
def compute_beta_jax(b, y_t, x_T, H, Xbeta):

    # 1. Define the function for a single loop iteration (the "scan body")
    # This function is defined inside so it can close over the non-iterating
    # variable `H`.
    def iteration(carry, x):
        # Unpack the carry state and the sliced inputs for this iteration
        Xs, ys = carry
        yt, xt, Xbeta_t = x

        # The logic is identical to the inside of your original for-loop
        na = jnp.isnan(yt)
        valid = ~na

        yt_valid = jnp.where(valid, yt, 0.0)
        Hx = H @ xt
        Hx_valid = jnp.where(valid, Hx, 0.0)

        r = yt_valid - Hx_valid
        # The `valid` mask needs to be broadcast to match Xbeta_t's shape
        Xbeta_t_masked = jnp.where(valid[:, None], Xbeta_t, 0.0)

        # Update the carry state
        ys_new = ys + Xbeta_t_masked.T @ r
        Xs_new = Xs + Xbeta_t_masked.T @ Xbeta_t_masked

        # We don't need to collect per-iteration results, so return None
        return (Xs_new, ys_new), None

    # 2. Prepare the initial state for the carry
    Xs_init = jnp.zeros((b, b))
    ys_init = jnp.zeros((b,))

    # 3. Prepare the data to be scanned over ('xs')
    # lax.scan iterates over the *first* axis. We need to rearrange our data
    # so that the time dimension is axis 0.

    # y_t has shape (N, T), transpose to (T, N)
    y_t_sliced = y_t.T

    # x_T has shape (N, T+1), we need slices from t+1, so we take [:, 1:]
    # and then transpose to (T, N)
    x_T_sliced = x_T[:, 1:].T

    # Xbeta has shape (N, b, T), move axis 2 to axis 0 -> (T, N, b)
    Xbeta_sliced = jnp.moveaxis(Xbeta, 2, 0)

    sliced_data = (y_t_sliced, x_T_sliced, Xbeta_sliced)

    # 4. Run the scan
    # The scan returns the final carry state and any collected outputs
    result, _ = jax.lax.scan(iteration, (Xs_init, ys_init), sliced_data)

    Xs_final, ys_final = result
    beta = jnp.linalg.solve(Xs_final, ys_final)

    return beta


@partial(jit, static_argnames=['block_p'])
def compute_s2e_jax(err, H, P_T, block_p):

    # 1. Prepare data for the time-scan. This is shared across all blocks.
    # We move the time axis to the front for lax.scan.
    # P_T shape: (q, q, T+1) -> we need t+1 slices, so we take [:,:,1:]
    # Resulting shape: (T, q, q)
    P_T_sliced = jnp.moveaxis(P_T[:, :, 1:], 2, 0)

    # err shape: (n, T) -> transpose to (T, n)
    err_sliced = err.T

    # 2. Define the function that computes s2e for a SINGLE block.
    # This is the function we will vectorize with `jax.vmap`.
    # It takes the start and end indices for its block as arguments.
    def compute_block_s2e(i_start, i_end):

        # --- Define the inner loop (scan over time) for this block ---

        # Slice H just once for this block.
        # Since block_p is static, i_start and i_end are constants during
        # compilation, so standard indexing is fine.
        H_p = H[i_start:i_end, :]  # Shape: (m_p, q)

        def iter_time(carry_acc, x_t):
            # Unpack the sliced data for the current time step t
            err_p_t, P_t_plus_1 = x_t

            # The logic is the same as your original compute_single_time
            nna = ~jnp.isnan(err_p_t)
            ep_valid = jnp.where(nna, err_p_t, 0.0)
            Hp_valid = H_p * nna[:, None]

            err_term = ep_valid @ ep_valid
            tr_term = jnp.trace(Hp_valid @ P_t_plus_1 @ Hp_valid.T)

            new_acc = carry_acc + err_term + tr_term
            return new_acc, None

        # --- Execute the logic for the single block ---

        # Slice the time-scannable err for this specific block
        err_p_sliced = err_sliced[:, i_start:i_end]  # Shape: (T, m_p)

        # The data to iterate over in the scan is a tuple of time-sliced arrays
        data_sliced = (err_p_sliced, P_T_sliced)

        # Run the scan over the time dimension starting from 0.0
        temp_p, _ = lax.scan(iter_time, 0.0, data_sliced)

        # Compute the denominator (number of non-NaNs for this block)
        nnaobs = jnp.sum(~jnp.isnan(err[i_start:i_end, :]))

        # Final result for this block, with a safeguard for division by zero
        return jnp.where(nnaobs > 0, temp_p / nnaobs, 0.0)

    # 3. Prepare inputs for vmap
    # We want to call compute_for_one_block for each pair of (start, end) indices.
    starts = block_p[:-1]
    ends = block_p[1:]

    # 4. Vectorize the single-block function over the start and end indices.
    # vmap will effectively call:
    # compute_block_s2e(starts[0], ends[0])
    # compute_block_s2e(starts[1], ends[1])
    # ... and so on, but in a single, efficient, compiled operation.

    s2e = jnp.asarray([compute_block_s2e(s, e) for s, e in zip(starts, ends)])

    return s2e


@partial(jit, static_argnames=['block_p', 'block_q', 'nvar', 'nlat', 'T'])
def compute_A2_jax(y_t, Xbeta, beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi):
    """
    Rewritten version of compute_A2 to work with jax.numpy and jax.jit,
    considering 'block_p', 'block_q', 'nvar', 'nlat', 'T', 'max_mdim_i' as static arguments.
    Fixes NonConcreteBooleanIndexError by padding to max_mdim_i and
    using jax.scipy.linalg.block_diag for static shapes.
    """
    # Precompute fixed effects and residuals for all data points
    # Shape (total_mdim, T)
    residual = y_t - jnp.einsum('mpn,p->mn', Xbeta, beta)

    # Initialize W (W will hold the results)
    W = jnp.zeros((nvar, nlat), dtype=y_t.dtype)

    # Loop over nvar (i) - nvar is a static argument
    for i in range(nvar):
        R_it = jnp.zeros((nlat, nlat), dtype=y_t.dtype)  # nlat is static
        g_it = jnp.zeros((1, nlat), dtype=y_t.dtype)     # nlat is static

        # Get the actual number of observations for the current block 'i'
        # mdim_i is static for a given `i` but can vary between `i` blocks.
        # max_mdim_i is the global maximum used for padding.
        mdim_i = block_p[i+1] - block_p[i]
        max_mdim_i = mdim_i

        # Loop over time steps T (t) - T is a static argument
        for t in range(T):
            # Extract the slice of y_t for the current block and time, then get NaN mask
            y_t_slice_full = y_t[block_p[i]:block_p[i+1], t]  # Shape (mdim_i,)
            # Boolean mask, shape (mdim_i,)
            nna_mask = ~jnp.isnan(y_t_slice_full)

            # Get the full residual for the current block and time
            residual_full_slice = residual[block_p[i]                                           :block_p[i+1], t]  # Shape (mdim_i,)

            # Mask and pad residual_full_slice
            # Multiply by mask to zero out invalid entries, then pad to max_mdim_i
            # This ensures ep_valid_padded always has shape (max_mdim_i,)
            ep_valid_padded = jnp.pad(residual_full_slice * nna_mask,  # (mdim_i,) * (mdim_i,)
                                      # Pad only the first dimension
                                      (0, max_mdim_i - mdim_i),
                                      constant_values=0.0)  # (max_mdim_i,)

            # --- Compute Psi_it (fixed shape) ---
            temp_padded_blocks = []
            # Assuming q_j_dim is constant, e.g., 1, as per common use with block_q
            # q_j_dim = block_q[j+1] - block_q[j]
            # This needs to be consistent, so let's use the first block_q diff for column padding.
            # Or ensure block_q always yields same q_j_dim for consistent Phi slicing.
            # Assuming block_q[j+1]-block_q[j] is always the same for all j (e.g. 1)
            # as in your example: block_q = jnp.array([0, 1, 2, nlat]) implies q_j_dim=1.
            # Get the dimension of a single q block
            q_j_dim = block_q[1] - block_q[0]

            for j in range(nlat):  # nlat is static
                # Current block of Phi, shape (mdim_i, q_j_dim)
                phi_block_current = Phi[block_p[i]
                    :block_p[i+1], block_q[j]:block_q[j+1]]

                # Apply mask by multiplying (zeros out rows corresponding to NaNs)
                # nna_mask[:, jnp.newaxis] broadcasts the mask to all columns of phi_block_current
                phi_block_masked = phi_block_current * nna_mask[:, jnp.newaxis]

                # Pad this masked block to (max_mdim_i, q_j_dim) for static shapes
                padded_phi_block = jnp.pad(phi_block_masked,
                                           ((0, max_mdim_i - mdim_i), (0, 0)),
                                           constant_values=0.0)
                temp_padded_blocks.append(padded_phi_block)

            # Construct the block diagonal matrix using jax.scipy.linalg.block_diag
            # If each q_j_dim is 1, then the total columns is nlat.
            # The total rows will be sum of rows from each block, i.e., nlat * max_mdim_i.
            # So, Psi_it now has shape (nlat * max_mdim_i, nlat) (if q_j_dim=1 for all j)
            Psi_it = jax.scipy.linalg.block_diag(*temp_padded_blocks)

            # --- Update g_it ---
            kron_terms = []
            for j in range(nlat):
                # Use max_mdim_i directly for jnp.eye to ensure static shape
                # This identity matrix now has a fixed size.
                # Shape (max_mdim_i, max_mdim_i)
                tt = jnp.eye(max_mdim_i, dtype=y_t.dtype)

                # kron_term_val_j shape: (max_mdim_i, nlat * max_mdim_i)
                kron_term_val_j = jnp.kron(ei_jax(j, nlat), tt)

                # Check for compatibility between ldim and nlat for Psi_it @ x_T
                # This is an implicit assumption from your original code's structure.
                # Shape: (nlat * max_mdim_i, 1)
                Psi_times_xT = Psi_it @ x_T[:, [t+1]]

                # Product shape: (max_mdim_i, nlat * max_mdim_i) @ (nlat * max_mdim_i, 1) = (max_mdim_i, 1)
                kron_terms.append(kron_term_val_j @ Psi_times_xT)

            # Concatenate these (max_mdim_i, 1) vectors horizontally to form (max_mdim_i, nlat) matrix
            combined_H_times_x = jnp.concatenate(
                kron_terms, axis=1)  # Shape: (max_mdim_i, nlat)

            # Add to g_it: ep_valid_padded.T is (1, max_mdim_i)
            # Result: (1, max_mdim_i) @ (max_mdim_i, nlat) = (1, nlat) -> Matches g_it's shape
            g_it += ep_valid_padded[jnp.newaxis, :] @ combined_H_times_x

            # --- Update R_it ---
            # Common term for R_it update, shape: (ldim, ldim)
            x_T_x_T_P = x_T[:, [t+1]] @ x_T[:, [t+1]].T + P_T[:, :, t+1]

            # Nested loops for j and k
            for j_idx in range(nlat):
                for k_idx in range(nlat):
                    # Use max_mdim_i directly for jnp.eye for static shape
                    tt_jk = jnp.eye(max_mdim_i, dtype=y_t.dtype)
                    # Kron term: shape (nlat * max_mdim_i, nlat * max_mdim_i)
                    kron_op_jk = jnp.kron(
                        ei_jax(j_idx, nlat).T @ ei_jax(k_idx, nlat), tt_jk)

                    # Calculate the term for trace: Psi_it.T @ kron_op_jk @ Psi_it @ (x_T @ x_T.T + P_T)
                    # Psi_it.T is (nlat, nlat * max_mdim_i)
                    # kron_op_jk is (nlat * max_mdim_i, nlat * max_mdim_i)
                    # Psi_it is (nlat * max_mdim_i, nlat)
                    # x_T_x_T_P is (ldim, ldim) -> (nlat, nlat) based on assertion

                    # The full product chain:
                    # (nlat, nlat * max_mdim_i) @ (nlat * max_mdim_i, nlat * max_mdim_i)
                    # -> (nlat, nlat * max_mdim_i) @ (nlat * max_mdim_i, nlat)
                    # -> (nlat, nlat) @ (nlat, nlat)
                    # -> (nlat, nlat)
                    term_to_trace = Psi_it.T @ kron_op_jk @ Psi_it @ x_T_x_T_P

                    R_it = R_it.at[j_idx, k_idx].add(jnp.trace(term_to_trace))

        # Solve linear system for W row
        w_i_row = jnp.linalg.solve(R_it, g_it.T).reshape(nlat,)

        # Update W (immutable operation using .at[].set())
        W = W.at[i, :].set(w_i_row)

    return W

# Helper function ei for JAX


def ei_jax(i, dim):
    """
    Creates a one-hot row vector for JAX.
    """
    return jax.nn.one_hot(i, dim, dtype=jnp.float32).reshape(1, dim)

# Apply jit with static_argnames
# Note: max_mdim_i is added back to static_argnames and function parameters


# %% [Fully Scipy] Argmin problem

# This new function encapsulates the part you want to run outside of JAX.


def minf(params, est_covList, T, Omega):
    ks = np.exp(params)

    # Compute the precision and the logdetQ (sparse matrix)
    invQ = [fcov.precision(rescale=ki)[:fcov.n_inner_points, :fcov.n_inner_points]
            for fcov, ki in zip(est_covList, ks)]

    invQ = sp.block_diag(invQ, format='csc')  # csr

    # Compute the likelihood
    logdet_invQ = logdetSparse(invQ)

    fun = -T * logdet_invQ + np.trace(invQ @ Omega)
    # print(ks, fun)

    return fun


def logdetSparse(Q):
    # Perform LU decomposition of the sparse matrix
    lu = sp.linalg.splu(Q)

    # Extract diagonal elements from L and U
    diagL = lu.L.diagonal().astype(np.complex128)
    diagU = lu.U.diagonal().astype(np.complex128)

    # Compute the log-determinant
    logdet = np.log(np.abs(diagL)).sum() + np.log(np.abs(diagU)).sum()

    return logdet


# %% Get initial values

@partial(jax.jit, static_argnums=(2, 3, 4))
def getInitialValues(y_t, Xbeta, block_p, block_q, T):
    """
    Computes initial parameter values for a model using JAX.

    Args:
        key (jax.random.PRNGKey): The random key for any stochastic operations.
        y_t (jnp.ndarray): The target variable array of shape (ni, T).
        Xbeta (jnp.ndarray): The feature array of shape (ni, b, T).
        block_p (tuple or list): Static list defining blocks for variables.
        block_q (tuple or list): Static list defining blocks for latent factors.
        T (int): Static integer for the number of time steps.

    Returns:
        A tuple of estimated initial values:
        (est_beta, est_s2, est_f, est_x0, est_Sigma0, est_A)
    """
    nvar = len(block_p) - 1
    nlat = len(block_q) - 1  # consider also the alpha

    # --- beta (OLS) ---

    # In JAX, arrays are immutable. Use jnp.where to replace nan/inf values
    # instead of in-place assignment.
    Yt_clean = jnp.nan_to_num(y_t)

    # Reshape X to (ni, T, b)
    Xr = Xbeta.transpose(0, 2, 1)  # (ni, T, b)

    # Compute Xs and ys efficiently using Einstein summation convention
    # This part is identical in syntax to NumPy.
    Xs = jnp.einsum('bij,bik->jk', Xr, Xr)
    ys = jnp.einsum('bij,bi->j', Xr, Yt_clean)

    # Solve for b using jnp.linalg.solve for numerical stability
    est_beta = jnp.linalg.solve(Xs, ys)

    # --- Mean variance of the residuals ---
    # Vectorize the residual calculation instead of using a Python loop.
    # 'ibt,b->it' means: multiply (ni, b, T) with (b,) -> result (ni, T)
    predicted_y = jnp.einsum('ibt,b->it', Xbeta, est_beta)
    res = Yt_clean - predicted_y

    # --- s2 measurement error [1 * p] ---
    # A Python loop over static values (from block_p) is acceptable and will
    # be "unrolled" by the JIT compiler.
    var_res_list = []
    for p in range(nvar):
        # Slicing and computing variance
        var_res_list.append(jnp.var(res[block_p[p]:block_p[p+1]]))

    var_res = jnp.stack(var_res_list)
    est_s2 = var_res * 0.2

    # --- est_A (Loading Matrix) ---
    # Create the matrix without loops using JAX's functional update syntax.
    diag_vals = jnp.sqrt(var_res * 0.8)
    diag_size = min(nvar, nlat)
    # Create a zero matrix and set the diagonal elements
    est_A = jnp.zeros((nvar, nlat))
    est_A = est_A.at[jnp.arange(diag_size), jnp.arange(
        diag_size)].set(diag_vals[:diag_size])

    # --- Random Initial Values ---
    # JAX requires explicit handling of random number keys.
    # Split the main key for each separate random operation.
    # key, f_key = jax.random.split(key)

    # est_f: Sorted initial values
    # rand_vals = jax.random.uniform(f_key, shape=(nlat,), minval=0.8, maxval=0.9)
    est_f = jnp.repeat(0.8, nlat)  # jnp.flip(jnp.sort(rand_vals))

    # est_x0: Initial state
    est_x0 = jnp.zeros((block_q[-1],))

    # est_Sigma0: Initial state covariance
    est_Sigma0 = 10 * jnp.eye(block_q[-1])

    return est_beta, est_s2, est_f, est_x0, est_Sigma0, est_A

# %% Utils functions EM


def buildRF(s2error, flatent, pdim, qdim):
    rdiag = sp.diags(np.repeat(s2error, list(pdim)),
                     format='csr', dtype=np.float64)
    fdiag = sp.diags(np.repeat(flatent, qdim), format='csr', dtype=np.float64)
    return rdiag, fdiag


def buildRF_dense(s2error, flatent, pdim, qdim):
    """
    Builds dense diagonal matrices in JAX.

    This is the recommended approach for a direct, easy-to-use replacement.

    Args:
        s2error: 1D array of error variances.
        flatent: 1D array of latent factor values.
        pdim: Tuple of block dimensions for s2error (static for JIT).
        qdim: Tuple of block dimensions for flatent (static for JIT).

    Returns:
        A tuple of two dense JAX diagonal matrices.
    """
    # 1. Create the full diagonal vector by repeating elements
    rdiag_vec = jnp.repeat(s2error, repeats=jnp.array(pdim)
                           ).astype(dtype=jnp.float32)
    fdiag_vec = jnp.repeat(flatent, repeats=jnp.array(qdim)
                           ).astype(dtype=jnp.float32)

    # 2. Create the dense diagonal matrices from the vectors
    rdiag_matrix = jnp.diag(rdiag_vec)
    fdiag_matrix = jnp.diag(fdiag_vec)

    return rdiag_matrix, fdiag_matrix


# dense matrix
def buildBasis_list(points, hmesh):
    nvar = len(points)
    nlat = len(hmesh)

    basis = []  # list of the rows x columns matrices
    notfindInx = []
    for p in range(nvar):
        hrow = []
        notfindInxRow = []
        for q in range(nlat):
            # This function works iif the cov class is already defined
            # Compute basis between vertex and gird_obs point [m x q]
            countij, notfindInxij, hij = hmesh[q]._compute_basis(
                points[p])

            hij = normalize_rows_sparse(hij[:, :hmesh[q].n_inner_points])
            notfindInxRow.append(notfindInxij)

            # conver into coo format
            hij = jnp.asarray(hij.toarray(), dtype=jnp.float32)

            # Append the sub matrices
            hrow.append(hij)

        notfindInx.append(notfindInxij)  # append not find index
        basis.append(hrow)  # [n_i x qsize ]

    return basis


def buildH_dense(A, basis):

    nvar, nlat = A.shape

    Phi = []  # list of the rows x columns matrices
    for p in range(nvar):
        hrow = []
        for q in range(nlat):
            hrow.append(A[p, q] * basis[p][q])
        Phi.append(hrow)

    return jnp.block(Phi)


def normalize_rows_sparse(sparse_mat):
    """
    Given a SciPy sparse matrix sparse_mat (CSR, CSC, etc.), return a new sparse matrix
    where each row sums to 1. Rows that originally sum to 0 will remain all zeros.
    """
    if not sp.isspmatrix(sparse_mat):
        raise ValueError("Input must be a SciPy sparse matrix.")

    # 1. Compute the sum of each row. The result is a (n_rows, 1) matrix, so we flatten to 1D.
    row_sums = np.array(sparse_mat.sum(axis=1)).ravel()  # shape = (n_rows,)

    # 2. Build an array of inverse sums, with zeros where row_sums == 0
    inv_sums = np.zeros_like(row_sums, dtype=np.float64)
    nonzero_mask = (row_sums != 0)
    inv_sums[nonzero_mask] = 1.0 / row_sums[nonzero_mask]

    # 3. Create a sparse diagonal matrix D so that D[i, i] = 1 / row_sums[i]
    #    Rows with row_sums == 0 get D[i,i] = 0, so that D @ row_i = 0 row.
    D = sp.diags(inv_sums)

    # 4. Multiply D @ sparse_mat; each row_i of the result is row_i of sparse_mat divided by row_sums[i].
    normalized = D.dot(sparse_mat)

    return normalized
# fix H row functions


def fixH_row(m_points, H, hmeshi, nfIndex, knn=2):

    dist = hmeshi.get_distance(points=m_points)

    if sp.issparse(H):
        H = H.toarray()

    # iterative solution
    for i in nfIndex:

        # find the nearest latent points
        nearest_pt = np.argsort(dist[i, :])[1:knn+1]

        # fix the weight
        wi = H[i, :].sum()

        H[i, nearest_pt] += (1 - wi)/knn

    return sp.csr_matrix(H)


# fix H columns functions
def fixH_column(m_points, H, hmeshi, thr=1e-1, knn=2):

    dist = hmeshi.get_distance(points=m_points)

    if sp.issparse(H):
        H = H.toarray()

    # iterative solution
    while True:
        inx = np.array(H[:, :hmeshi.n_inner_points].sum(
            axis=0) <= thr).reshape(-1,)  # of course not the boundary

        fixindex = inx.nonzero()[0]
        # print(fixindex)

        if len(fixindex) == 0:
            # return
            break
        else:
            # fix procidure
            for i in fixindex:

                # find the nearest points
                nearest_pt = np.argsort(dist[:, i])[1:knn+1]

                # multiplay by 90%
                H[nearest_pt, :] = H[nearest_pt, :]*0.8

                # add the 0.1 quantity
                H[nearest_pt, i] = H[nearest_pt, i] + 0.2

    return sp.csr_matrix(H)

# %% mesh helper


def compute_angles(vertex, triangle):
    A, B, C = vertex[triangle]

    def angle(a, b, c):
        ab = np.linalg.norm(b - a)
        ac = np.linalg.norm(c - a)
        bc = np.linalg.norm(c - b)
        cos_theta = (ab**2 + ac**2 - bc**2) / (2 * ab * ac)
        # Convert to degrees
        return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

    return np.array([angle(A, B, C), angle(B, C, A), angle(C, A, B)])


# Compute triangle area
def compute_area(vertex, triangle):
    A, B, C = vertex[triangle]
    return 0.5 * abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))


def isinner(boundary_polygon, points):
    return boundary_polygon.intersects(
        [Point(p) for p in points])  # or contains + polygin


def get_boundary_index(boundary_polygon, points):
    bound_x, bound_y = boundary_polygon.boundary.coords.xy
    bound_x, bound_y = bound_x[:-1], bound_y[:-1]

    # boundary_vertices = []
    # for pt in:
    #     inx = np.where(np.bitwise_and(
    #         vertex[:, 0] == pt[0], vertex[:, 1] == pt[1]))

    #     # if len(inx) == 1:
    #     boundary_vertices.append(inx)

    temp = np.bitwise_and(
        np.isin(points[:, 0], bound_x), np.isin(points[:, 1], bound_y))

    return np.nonzero(temp)[0]


def reduce_vertex_count_hard(vertex, boundary_polygon, max_vertices):

    # compute the boundary poligon
    # boundary_vertices = hull.vertices  # index of boundary vertices
    # Ger the inner vertex (remove the nodes only from the inner vertices)
    tri = None
    inner_vertex = vertex[isinner(boundary_polygon, vertex)]

    while len(inner_vertex) > max_vertices:

        # Compute the triangolarisation
        tri = Delaunay(vertex)
        triangles = tri.simplices
        vertex = tri.points

        # Ger the inner vertex (remove the nodes only from the inner vertices)
        inner_vertex = vertex[isinner(boundary_polygon, vertex)]

        # Get the boundary index
        boundary_index = get_boundary_index(boundary_polygon, vertex)
        if len(boundary_index) == len(inner_vertex):
            return Delaunay(vertex)

        # compute the area and sort
        areas = np.array([compute_area(vertex, t) for t in triangles])
        sort_area_inx = np.argsort(areas)

        # Find a vertex to remove that is NOT on the boundary
        inx = 0
        vertex_to_remove = None

        while vertex_to_remove is None:
            candidate_vertices = triangles[sort_area_inx[inx]]

            for v in candidate_vertices:
                if v not in boundary_index:
                    vertex_to_remove = v
                    break

            if vertex_to_remove is None:
                inx += 1

        # Get the new vertices
        vertex = np.delete(vertex, vertex_to_remove, axis=0)

        # print(len(inner_vertex), len(boundary_index),
        #      len(vertex), vertex_to_remove)

    if len(inner_vertex) <= max_vertices:
        tri = Delaunay(vertex)
        # triangles = tri.simplices
        # vertex = tri.points

    return tri


def laplacian_smoothing(delaunay, boundary_polygon, max_iterations=10, angle_thr=5, alpha=0.2, verbose=True):
    """Smooth internal points by averaging with neighbors, keeping boundary points fixed."""

    # Convex hull of observation
    # Points = all points (inner + outer)

    triangles = delaunay.simplices
    vertex = delaunay.points

    # Get boundary vertex indices (not coordinates)
    boundary_index = get_boundary_index(boundary_polygon, vertex)

    # Create boundary polygon
    inner = isinner(boundary_polygon, vertex)
    new_points = vertex.copy()

    iteration = 0
    while True:

        min_angle = np.round(np.min([np.min(compute_angles(vertex, t))
                                     for t in delaunay.simplices]), 2)

        if verbose:
            print(iteration, min_angle, angle_thr)

        if min_angle >= angle_thr or iteration > max_iterations:
            break
        else:
            iteration += 1
            for i in range(len(vertex)):
                if i in boundary_index:
                    continue  # Skip boundary points

                neighbors = set()
                for t in triangles:
                    if i in t:
                        neighbors.update(t)

                neighbors.remove(i)  # Remove self from neighbor list

                if len(neighbors) > 0:
                    avg_position = np.mean(vertex[list(neighbors)], axis=0)
                    current_position = (1 - alpha) * \
                        vertex[i] + alpha * avg_position

                    # check if it is insede the polygon
                    flag = isinner(boundary_polygon,
                                   current_position.reshape(1, 2))
                    if inner[i] and not flag:
                        continue

                    if not inner[i] and flag:
                        continue

                    new_points[i] = current_position
        vertex = new_points

    return vertex, min_angle
