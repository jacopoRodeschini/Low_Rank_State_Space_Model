#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:41:10 2025

@author: jacopo
@title: test_std
"""

# %% import library
from jax.scipy.linalg import solve
from jax import checkpoint
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri

import jax
from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import pickle
import gstools as gs
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

# from scipy.linalg import block_diag
from jax.scipy.linalg import block_diag
from gstools.covmodel import Matern
from scipy.optimize import minimize
import time
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import geopandas as gpd
from patsy import dmatrices
import geopandas
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from scipy.spatial import ConvexHull
from shapely.ops import unary_union


from utils import spdeAppoxCov
from utils import filter_smw_nan, smoother_smw, computeExpectedValues
from utils import grid
import argparse


# main function
def run_case_study(lowrank):

    # %% import dataset
    print("Import dataset")

    # Import ancheAgrimonia dataset
    filepath = './output/empirical_application.csv'

    grins = pd.read_csv(filepath, na_values=[
        "      NaN", "", "NA", "null", "-"], keep_default_na=True)

    grins['Time'] = pd.to_datetime(grins['time'], format="%Y-%m-%d")

    # Create Point(lat,lon) for each observation
    ct = np.array([grins.Longitude.to_numpy(), grins.Latitude.to_numpy()]).T
    grins['geometry'] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

    # create geopandas dataset
    grins = gpd.GeoDataFrame(grins, crs=4326)

    # remove the outliers
    grins.rename(columns={"mean_PM2.5": "mean_PM25"}, inplace=True)
    thr10 = grins.mean_PM10.quantile(0.999)
    thr25 = grins['mean_PM25'].quantile(0.999)

    grins.loc[grins.mean_PM10 > thr10, 'mean_PM10'] = np.nan
    grins.loc[grins['mean_PM25'] > thr25, 'mean_PM25'] = np.nan

    # compute the covariance
    covariance = grins[['mean_PM10', 'mean_PM25']].dropna().corr()

    # %% Import italian shapefile
    print("Import shapefile")

    shapepath = './shapefile/Reg01012024_g_WGS84.shp'

    sh_italy = geopandas.read_file(shapepath)
    sh_italy.to_crs(4326, inplace=True)

    # compute only the continetal italian (connected mesh)
    ct_italy = sh_italy.copy()

    # get the largest polygon
    ct_italy['geometry'] = ct_italy['geometry'].apply(
        lambda geo: getLargestPoly(geo))
    ct_italy.plot()

    # filter shape file
    italy_union = unary_union(ct_italy.geometry)
    inner_buffer = italy_union.buffer(-0.15)

    # make a plot
    inner_buffer_gs = gpd.GeoSeries([inner_buffer], crs=ct_italy.crs)

    # %% filter the grins dataset over the continental shapefile

    # get the unique points
    unique_points = grins.geometry.drop_duplicates().reset_index(drop=True)
    unique_gdf = gpd.GeoDataFrame(geometry=unique_points, crs=grins.crs)

    # filter over the continental italy
    contained = gpd.sjoin(unique_gdf, ct_italy, predicate='within', how='left')
    unique_gdf['inside'] = ~contained['index_right'].isna()

    geom_to_inside = dict(
        zip(unique_gdf.geometry, unique_gdf['inside']))  # lookup table

    grins['inside'] = grins.geometry.map(geom_to_inside)

    # filter
    grins = grins[grins['inside']]

    # %% build dataframe
    print("Build the covariates matrices")

    nvar, points, gridList, ndim, pdim, block_p, T = buildObservationGrid(
        grins, ['mean_PM10 ~ 1 + Altitude + ERA5Land_t2m + ERA5Land_rh + ERA5Land_windspeed',
                'mean_PM25 ~ 1 + Altitude + ERA5Land_t2m + ERA5Land_rh + ERA5Land_windspeed'])

    # Y - Built the response variable [(n1 + n2 + ... + nq) x T] = [N x T]
    Ylist_original = [grid.y for grid in gridList]

    # applay the log transofrmation (natural log) [positive prediction]
    Ylist = Ylist_original

    Xlist = [grid.X for grid in gridList]

    # %% Test and train dataset
    print("Split test and train dataset")

    # fix the seed the get every times the same station
    hull = [ConvexHull(pt) for pt in points]
    boundary_polygon = [Polygon(h.points[h.vertices])
                        for h in hull]  # .buffer(0.01)
    boundary_vertices = [h.vertices for h in hull]

    # Train and test random set (CV pourples)
    item = [np.arange(len(pt)) for pt in points]
    # item = [np.delete(it, bd_pt) for it, bd_pt in zip(
    #     item, boundary_vertices)]  # [boundary always on the train]

    # buffer points always on the train ()
    only_train = []
    for i in range(len(points)):
        geometries = [Point(xy) for xy in points[i]]

        pt_gdf = gpd.GeoDataFrame(geometry=geometries, crs=ct_italy.crs)
        valid = ~pt_gdf.geometry.within(inner_buffer)

        temp = np.arange(0, len(points[i]), 1)[valid]
        only_train.append(temp)

        item[i] = np.delete(item[i], temp)

    # get the random split of the dataset
    np.random.seed(1234)
    choice = [np.random.choice(it, size=(len(it),), replace=False)
              for it in item]

    # number of triaining points
    ntest = [int(np.ceil(0.1 * len(pt)))
             for pt in points]  # ntrain = ntrain + border
    itest = [ch[:nt] for ch, nt in zip(choice, ntest)]

    itrain = [np.hstack((ch[nt:], bd_pt))
              for ch, nt, bd_pt in zip(choice, ntest, only_train)]

    # fix vertex = observed locations
    # vertex = coords[itrain, :]

    # plot the resutint validation schemes
    fix, ax = plt.subplots(1, 2)
    for i in range(len(ax)):
        inner_buffer_gs.plot(ax=ax[i])
        ct_italy.boundary.plot(ax=ax[i], linewidth=0.5)
        ax[i].plot(points[i][itrain[i], 0], points[i]
                   [itrain[i], 1], 'gx', alpha=0.5)
        ax[i].plot(points[i][itest[i], 0], points[i][itest[i], 1], 'mx')

    # %% Filter train and test
    points_train = [pt[index, :] for pt, index in zip(points, itrain)]
    points_test = [pt[index, :] for pt, index in zip(points, itest)]

    Y_train_list = [yi[index, :] for yi, index in zip(Ylist, itrain)]
    Xbeta_train_list = [xi[index, :, :] for xi, index in zip(Xlist, itrain)]

    Y_test_list = [yi[index, :] for yi, index in zip(Ylist, itest)]
    Xbeta_test_list = [xi[index, :, :] for xi, index in zip(Xlist, itest)]

    Y_train = jnp.vstack(Y_train_list)
    Xbeta_train = block_diag_3D(Xbeta_train_list)

    Y_test = np.vstack(Y_test_list)
    Xbeta_test = block_diag_3D(Xbeta_test_list)

    tlag = int(Y_train.shape[1]/2)

    # %% Create the mesh for the latent grid (lowrank = 50%) [unidimensional]
    print("Create the mesh")

    nlat = 2
    if nlat > len(points_train):
        print("dim latent> process")

    # 107       # Define the maxium number of vertices for the latent mesh
    if nlat == len(points_train):
        points_mesh = points_train
    else:
        points_mesh = [np.unique(np.vstack(points_train), axis=0)]

    hull = [ConvexHull(pt) for pt in points_mesh]
    boundary_polygon = [Polygon(h.points[h.vertices])
                        for h in hull]  # .buffer(0.01)
    boundary_vertices = [h.vertices for h in hull]

    # lowrank = 1  # [1, 0.75, 0.5, 0.25]
    max_vertices = [int(np.ceil(lowrank * len(vertex)))
                    for vertex in points_mesh]  # train

    max_iterations = 10     # Avoid infinite loops laplace algorithm
    angle_thr = 5           # Angle bounded away from 0 threshold

    # Compute the outer points
    delta = 0.5
    x = [np.linspace(start=h.min_bound[0]-delta,
                     stop=h.max_bound[0]+delta, num=15) for h in hull]
    y = [np.linspace(start=h.min_bound[1]-delta,
                     stop=h.max_bound[1]+delta, num=15) for h in hull]

    outer_grid = [np.meshgrid(xi, yi) for xi, yi in zip(x, y)]

    outer_points = [np.vstack(
        (gr[0].flatten(), gr[1].flatten())).T for gr in outer_grid]  # Mesh vertex grid

    # remove the outer point within the polygon
    index = [bd_poly.contains([Point(p) for p in ot_points])
             for bd_poly, ot_points in zip(boundary_polygon, outer_points)]
    outer_points = [ou_points[~inx]
                    for ou_points, inx in zip(outer_points, index)]

    # compure the all_points (the inner points index and the boundary coordinates)
    all_points = [np.vstack((vertex, ou_points))
                  for vertex, ou_points in zip(points_train, outer_points)]
    # inner = isinner(boundary_polygon, all_points)  # or contains + polygin

    # Reduce the vertex count and compute a valid delanuay
    delaunay = [reduce_vertex_count_hard(
        all_pt, bd_poly, m_v) for all_pt, bd_poly, m_v in zip(all_points, boundary_polygon, max_vertices)]

    # check the beginning min angle
    min_angle = [np.round(np.min([np.min(compute_angles(d.points, t))
                                 for t in d.simplices]), 2) for d in delaunay]

    # %% Laplacian smoothing
    print("Laplacian smoothing")
    #  (make the triangle more equilateral)
    angle_thr = 0.15

    sm_points = []
    fcov = []
    for i in range(len(delaunay)):

        new_points, min_angle = laplacian_smoothing(
            delaunay[i], boundary_polygon[i], max_iterations=20, angle_thr=angle_thr)

        sm_points.append(new_points)

        # build the covariance mesh
        inner = isinner(boundary_polygon[i], new_points)

        temp = spdeAppoxCov(new_points, delaunay[i].simplices, outer_index=~inner, add_boundary=True,
                            latlon=False, uniformRef=False, rescale=10)

        fcov.append(temp)

    # %% estimate the empirical covariance model
    print("Estimate the empirical covariance model")

    # list of the variables
    model = []
    for field, Xbeta, pt in zip(Y_train_list, Xbeta_train_list, points_train):
        # Get the field
        # field = agg_mean[res].to_numpy()

        field = np.nanmean(field, axis=1)
        inx = ~np.isnan(field)

        field = field[inx]
        Xbeta = Xbeta[inx, :, :].mean(axis=2)
        pt = pt[inx]

        # compute the residual
        n = len(field)
        P = Xbeta @ np.linalg.inv(Xbeta.T @ Xbeta) @ Xbeta.T
        res = (np.eye(n) - P) @ field

        # empirical variogram pm10
        bin_center, gamma = gs.vario_estimate((pt[:, 0], pt[:, 1]), res,
                                              bin_no=50,
                                              latlon=False, estimator='cressie')

        # fit the variogram with a stable model
        fit_model = gs.Matern(dim=2)
        fit_model.fit_variogram(
            bin_center, gamma, nugget=True, nu=1, anis=False)

        model.append(fit_model)

    # %% Estimate the model
    print("Estimate the model")

    pdim = jnp.array([len(i) for i in itrain])
    block_p = jnp.hstack((0, np.cumsum(pdim)))
    qdim = jnp.array([cov.n_inner_points for cov in fcov])
    block_q = jnp.hstack((0, np.cumsum(qdim)))

    rxy = 0.9
    vlat = jnp.array([mdl.var for mdl in model])

    # get some reasonable starting values
    beta0 = None  # Get by the ols
    # Initial estimate by semi-variogram (on the mean?)
    ks0 = jnp.array([np.sqrt(8)/mdl.len_scale for mdl in model])

    # A0 = np.array(vlat).reshape((2, 1))  # Initial mean covariance
    A0 = jnp.diag(vlat)
    A0 = A0.at[0, 1].set(jnp.sqrt(vlat).prod() * rxy)
    A0 = A0.at[1, 0].set(A0[0, 1])
    A0 = jnp.linalg.cholesky(A0).T

    s2e0 = jnp.array([mdl.nugget for mdl in model])  # the residual OLS rmse
    f0 = jnp.array([0.8, 0.8])  # Random initial guess

    # Estimate the model
    tlag = int(Y_train.shape[1]/2)

    est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_covList, est_A, nstat, y_hat, x_T, P_T, P_T_1, S11, S10, S00 = fit(
        Y_train[:, -tlag:], fcov, block_p, pdim, Xbeta_train[:,
                                                             :, -tlag:], points_train,
        beta0=beta0, s2e0=s2e0, f0=f0, A0=A0, ks0=ks0, x0=None, Sigma0=None,
        max_iter=100, tol_par=1e-4, tol_relat=1e-4, nstat=[])

    # df = pd.DataFrame(nstat)
    est_ks = jnp.array([f.rescale for f in est_covList], dtype=jnp.float32)

    # %% Compute the Standard deviation
    print("Compute the standard deviation")
    # Stack the ML Estimators (MLE)
    par = jnp.hstack((est_beta.flatten(), est_A.flatten(), est_s2e.flatten(), est_f.flatten(),
                     est_ks.flatten()))

    stack_dim = jnp.cumsum(jnp.array([0, len(est_beta), len(
        est_A.flatten()), len(est_s2e), len(est_f), len(est_ks)]))

    # compute basis
    basis = buildBasis_list(points_train, est_covList)
    stiff = [np.asarray(f.stiff.toarray()).copy() for f in est_covList]
    mass = [np.asarray(f.mass.toarray()).copy() for f in est_covList]
    ninner = jnp.array([f.n_inner_points for f in est_covList], dtype=jnp.int32)

    # tlag = 80
    # Compute the hessian function consideirng all the other parametrs fixed
    lik_fixed = partial(
        loglikelihood,
        stack_dim=stack_dim,           # Stack par dimensio
        basis=basis,
        y_t=Y_train[:, -tlag:],           # ensure JAX array
        Xbeta=Xbeta_train[:, :, -tlag:],
        x_T=x_T,
        P_T=P_T,
        S11=S11,
        S10=S10,
        S00=S00,
        x0=est_x0,
        Sigma0=est_Sigma0,
        pdim=pdim,
        qdim=qdim,
        stiff=stiff,
        mass=mass,
        ninner=ninner)

    ts = time.time()
    hesfun = jax.hessian(lik_fixed)
    IFisher = hesfun(par)
    jax.block_until_ready(IFisher)
    tdelta = time.time() - ts

    # Compute the Var-Cov matrix of the paramiters
    SigmaPar = jnp.linalg.inv(IFisher)

    # check if it is postive definte
    chol = jnp.linalg.cholesky(SigmaPar)

    # Compute the std
    std_par = np.sqrt(SigmaPar.diagonal())

    # %% Save the resuls (without the hessian evaluation)
    print("Save the results")

    output = {
        'nvar': nvar,
        'nlat': nlat,
        'pdim': pdim,
        'qdim': qdim,
        'pdim_test': [len(pt) for pt in points_test],
        'points_train': points_train,
        'points_test': points_test,
        'y_train': Y_train[:, -tlag:],
        'y_test': Y_test[:, -tlag:],
        'lowrank': lowrank,
        'y_name': [g._y_design_info.column_names for g in gridList],
        'Xbeta_train': Xbeta_train[:, :, -tlag:],
        'Xbeta_test': Xbeta_test[:, :, -tlag:],
        'x_name': [g._x_design_info.column_names for g in gridList],
        'mesh': fcov,
        'beta': est_beta,
        's2e': est_s2e,
        'f': est_f,
        'A': est_A,
        'ks': [f.rescale for f in est_covList],
        'x0': est_x0,
        'Sigma0': est_Sigma0,
        'nstat': nstat,
        'x_T': x_T,
        'P_T': P_T,
        'P_T_1': P_T_1,
        'S00': S00,
        'S10': S10,
        'S11': S11,
        'y_hat': y_hat,
        'y_hat_back': jnp.exp(y_hat),
        'Sigma_par': SigmaPar,
        'Std_par': std_par,
        'stack_par': par,
        'stack_dim': stack_dim,
    }

    # save the results
    filename = f'./output/LSSM_grins_2x2_R{int(lowrank*100)}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(output, f)


# %% Utils function

def getLargestPoly(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        # Return the polygon with the largest area
        # (itarable, key = scalar ordering function)
        return max(geometry.geoms, key=lambda p: p.area)
    else:
        return geometry  # Just in case


def buildObservationGrid(df, formula):

    nvar = len(formula)  # numer of the response variable
    gridList = [grid(df, f) for f in formula]

    T = [gr.T for gr in gridList]
    points = [gr.points for gr in gridList]

    # get dimnesion of each grid
    ndim = [grid.N for grid in gridList]
    block = np.hstack((0, np.cumsum(ndim)))

    return nvar, points, gridList, ndim, block[-1], block, T


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

# %% Function to evaluate the hessian matrix


def loglikelihood(par, stack_dim, basis, y_t, Xbeta, x_T, P_T, S11, S10, S00, x0, Sigma0, pdim, qdim, stiff, mass, ninner):
    beta0 = par[stack_dim[0]:stack_dim[1]]
    A0 = par[stack_dim[1]:stack_dim[2]].reshape((2, 2))
    s2e0 = par[stack_dim[2]:stack_dim[3]]
    f0 = par[stack_dim[3]:stack_dim[4]]
    ks0 = par[stack_dim[4]:stack_dim[5]]

    H = buildH_dense(A0, basis)  # dense
    invR, F = buildRF_dense(1/s2e0, f0, pdim, qdim)

    # get the precision matrix
    invSigma0 = np.linalg.inv(Sigma0)
    invQ = compute_invQ_jax(ks0, stiff, mass, ninner)

    Omega = S11 - S10 @ F.T - F @ S10.T + F @ S00 @ F.T

    # compute the observed likelihood
    logL = compute_logL(y_t, invR, H, invQ, x0, invSigma0,
                        Xbeta, beta0, x_T, P_T, Omega)
    jax.block_until_ready(logL)

    return logL

# positive likelihood


def compute_logL(y_t, invR, H, invQ, x0, invSigma0, Xbeta, beta, x_T, P_T, Omega):
    p, T = y_t.shape

    # --- Precompute log-dets ---
    logdet_invQ = jnp.linalg.slogdet(invQ)[1]

    # --- Likelihood of yt (sum over t) ---
    lik_yt = compute_logL_yt(y_t, invR, Xbeta, beta, H, x_T, P_T)

    # --- Likelihood of xt (sum over t) ---
    lik_xt = -T * logdet_invQ + jnp.trace(invQ @ Omega)

    return lik_yt + lik_xt


@jax.jit
def compute_logL_yt(y_t, invR, Xbeta, beta, H, x_T, P_T):
    """
    Computes the measurement log-likelihood using a JAX scan loop for efficiency.
    """
    p, T = y_t.shape

    def _scan_body(carry_log_lik, t):
        # This function processes a single time step t.
        # carry_log_lik: The accumulated log-likelihood from previous steps.
        # t: The current time step index.

        # 1. Get the data slices for the current time step
        y_slice = y_t[:, t]
        x_slice = x_T[:, t]
        P_slice = P_T[:, :, t]
        Xbeta_slice = Xbeta[:, :, t]

        # 2. Create the mask for observed (non-NaN) data
        observed_idx = ~jnp.isnan(y_slice)

        # Define the two branches for our conditional execution
        def true_fun(_):
            # --- This code runs if AT LEAST ONE value is observed ---

            # Create a diagonal masking matrix from the boolean vector
            # This is key: it keeps all operations in fixed p x p shape
            M = jnp.diag(observed_idx.astype(jnp.float32))

            # Mask the inverse covariance of the measurement error
            # This effectively selects the sub-matrix corresponding to observed data
            invR_t = M @ invR @ M

            # Calculate the log-determinant of the relevant sub-matrix.
            logdet_invR_t = jnp.linalg.slogdet(
                # Use `where` to avoid taking the determinant of a singular matrix
                # if some values are not observed. Fill with identity.
                jnp.where(M > 0, invR_t, jnp.eye(p))
            )[1]

            # Calculate the full-size residual
            residual_full = y_slice - Xbeta_slice @ beta - H @ x_slice
            # Mask the residual, setting missing entries to 0 so they don't contribute
            residual_t = jnp.where(observed_idx, residual_full, 0.0)

            # Calculate the quadratic form using the full-size matrices.
            quadratic_form = residual_t.T @ invR @ residual_t

            # Calculate the trace term similarly
            # H_t @ P_slice @ H_t.T
            H_t = M @ H
            trace_term = jnp.trace(H_t @ P_slice @ H_t.T)

            return -logdet_invR_t + quadratic_form + trace_term

        def false_fun(_):
            # --- This code runs if ALL values are missing ---
            # The contribution to the log-likelihood is zero.
            return 0.0

        # Use lax.cond to safely choose which branch to execute
        lik_t = lax.cond(jnp.any(observed_idx), true_fun,
                         false_fun, operand=None)

        # 4. Update the carry and return it for the next iteration
        new_carry = carry_log_lik + lik_t
        # The second element returned is for per-step outputs, which we don't need.
        return new_carry, None

    # Initial value for the log-likelihood accumulator
    initial_log_lik = 0.0

    # Run the scan operation over all time steps from 1 to T
    log_lik_yt, _ = lax.scan(_scan_body, initial_log_lik, jnp.arange(1, T+1))

    return log_lik_yt


def compute_invQ_jax(ks, stiff, mass, ninner):
    invQi = []
    # Loop over the parameters and matrices. A Python loop is fine here.
    for i in range(len(ks)):
        k = ks[i]

        current_mass = mass[i]
        current_stiff = stiff[i]
        current_inner = ninner[i]

        Cinv = jnp.diag(1.0 / current_mass.diagonal())

        K = (k**2) * current_mass + current_stiff

        sigma2k = jax.scipy.special.gamma(
            1.0) / (jax.scipy.special.gamma(2.0) * 4 * jnp.pi * (k**2))

        invQ = sigma2k * (K @ Cinv @ K)
        invQ = invQ[:current_inner, :current_inner]

        invQi.append(invQ)

    # 4. Use `jax.scipy.linalg.block_diag`
    return block_diag(*invQi)


# %% Fit the model


def fit(y_t, est_covList, block_p, pdim, Xbeta, points, beta0=None, s2e0=None,
        f0=None, A0=None, ks0=None, x0=None, Sigma0=None,
        max_iter=100, tol_par=1e-2, tol_relat=1e-3, nstat=[], verbose=True):

    # get global constant
    nvar = len(pdim)
    nlat = len(est_covList)

    # create the vector_parameter
    est_ks = jnp.array([cov.rescale for cov in est_covList])
    stiff = [f.stiff.toarray() for f in est_covList]
    mass = [f.mass.toarray() for f in est_covList]

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

    stiff, mass

    # Start EM iteration
    while flag:
        niter += 1

        # ---- build parametrised matrices
        tStart_iter = time.time()
        H = buildH_dense(est_A, basis)  # dense

        # R, F = buildRF(est_s2e, est_f, pdim, qdim)
        R, F = buildRF_dense(est_s2e, est_f, pdim, qdim)

        invQ = [jnp.asarray(fcov.precision()[:fcov.n_inner_points, :fcov.n_inner_points].toarray(), dtype=jnp.float32)
                for fcov in est_covList]

        Q = block_diag(
            *[jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=jnp.float32)) for mt in invQ])
        # Q = compute_Q_jax(est_ks, stiff, mass, tuple(ninner.tolist()))

        # ---- E step
        y_hat, x_t, x_T, P_T, P_T_1, S11, S10, S00, logL_cur, tdelta_Edet = E_step(
            y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta)

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

    return est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_covList, est_A, nstat, y_hat, x_T, P_T, P_T_1, S11, S10, S00

# %% E step


def E_step(y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta):

    # python
    tStart = time.time()
    x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL = filter_smw_nan(
        y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta)
    jax.block_until_ready(x_t)
    tdelta_kf = time.time() - tStart

    tStart = time.time()
    x_T, P_T, P_T_1 = smoother_smw(
        H, F, x_t, P_t, K, x_t_1, P_t_1, invP_t_1)
    jax.block_until_ready(x_T)
    tdelta_sm = time.time() - tStart

    # Output Eq: (7a, 7b, 7c, 7d, 7e)
    tStart = time.time()
    y_hat, S11, S10, S00 = computeExpectedValues(
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

    tStart = time.time()
    beta = compute_beta_jax(b, y_t, x_T, H, Xbeta)
    jax.block_until_ready(beta)
    tdelta_beta = time.time() - tStart

    # compute the s2e
    tStart = time.time()
    s2e = compute_s2e_jax(err, H, P_T, tuple(block_p.tolist()))
    jax.block_until_ready(s2e)
    tdelta_s2e = time.time() - tStart

    # update A
    tStart = time.time()

    est_A = compute_A2_jax(y_t, Xbeta, beta, x_T, P_T,
                           tuple(block_p.tolist()), tuple(block_q.tolist()),  nvar, nlat, T, ldim, Phi)
    jax.block_until_ready(est_A)
    tdelta_A = time.time() - tStart

    # update the covraiance object
    tStart = time.time()
    Omega = S11 - S10 @ F.T - F @ S10.T + F @ S00 @ F.T
    par0 = jnp.log(
        jnp.array([fcov.rescale for fcov in est_covList], dtype=jnp.float32))

    opt = minimize(minf, par0, args=(est_covList, T, Omega), method='BFGS',  # method='L-BFGS-B'
                   tol=1e-3, jac=False, options={'maxiter': 100})

    tdelta_ks = time.time() - tStart

    # update the initial state mu and variance
    x0 = x_T[:, 0]
    Sigma0 = P_T[:, :, 0]

    tdelta = jnp.array([tdelta_beta, tdelta_s2e, tdelta_f,
                       tdelta_ks, tdelta_A], dtype=jnp.float32)
    return beta, s2e, est_f, x0, Sigma0, est_covList, est_A, tdelta

# %% [JAX] updating formula


@partial(jit, static_argnames=['b'])
def compute_beta_jax(b, y_t, x_T, H, Xbeta):

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

    # y_t has shape (N, T), transpose to (T, N)
    y_t_sliced = y_t.T

    # x_T has shape (N, T+1), we need slices from t+1, so we take [:, 1:]
    # and then transpose to (T, N)
    x_T_sliced = x_T[:, 1:].T

    # Xbeta has shape (N, b, T), move axis 2 to axis 0 -> (T, N, b)
    Xbeta_sliced = jnp.moveaxis(Xbeta, 2, 0)

    sliced_data = (y_t_sliced, x_T_sliced, Xbeta_sliced)

    # The scan returns the final carry state and any collected outputs
    result, _ = jax.lax.scan(iteration, (Xs_init, ys_init), sliced_data)

    Xs_final, ys_final = result
    beta = jnp.linalg.solve(Xs_final, ys_final)

    return beta


@partial(jit, static_argnames=['block_p'])
def compute_s2e_jax(err, H, P_T, block_p):

    P_T_sliced = jnp.moveaxis(P_T[:, :, 1:], 2, 0)

    # err shape: (n, T) -> transpose to (T, n)
    err_sliced = err.T

    def compute_block_s2e(i_start, i_end):

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

    starts = block_p[:-1]
    ends = block_p[1:]

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
            residual_full_slice = residual[block_p[i]:block_p[i+1], t]  # Shape (mdim_i,)

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
                phi_block_current = Phi[block_p[i]:block_p[i+1], block_q[j]:block_q[j+1]]

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

    invQ = sp.block_diag(invQ, format='csr')

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

    # est_f: Sorted initial values
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

    row_sums = np.array(sparse_mat.sum(axis=1)).ravel()  # shape = (n_rows,)

    inv_sums = np.zeros_like(row_sums, dtype=np.float64)
    nonzero_mask = (row_sums != 0)
    inv_sums[nonzero_mask] = 1.0 / row_sums[nonzero_mask]

    D = sp.diags(inv_sums)

    normalized = D.dot(sparse_mat)

    return normalized

# %% Mesh utils functions


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

    temp = np.bitwise_and(
        np.isin(points[:, 0], bound_x), np.isin(points[:, 1], bound_y))

    return np.nonzero(temp)[0]


def reduce_vertex_count_hard(vertex, boundary_polygon, max_vertices):

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

    return tri


def laplacian_smoothing(delaunay, boundary_polygon, max_iterations=10, angle_thr=5, alpha=0.2, verbose=True):
    """Smooth internal points by averaging with neighbors, keeping boundary points fixed."""

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


# %% main section
# main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run case study with a parameter.")
    parser.add_argument("lowrank", type=float,
                        help="Lowrank parameter hat to pass to run_case_study")
    args = parser.parse_args()
    lowrank = args.lowrank

    # run the case study
    print(f"Run case study with lowrank={lowrank}")
    run_case_study(lowrank)
