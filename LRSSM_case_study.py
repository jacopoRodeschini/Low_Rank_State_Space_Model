#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 10:18:19 2025

@author: jacopo
"""

# %% import library
import jax.numpy as jnp
import pickle
import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import jax
import time


import pandas as pd
import geopandas as gpd
import geopandas
from scipy.spatial import ConvexHull
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon

from pathlib import Path

from utils import getLargestPoly, buildObservationGrid, block_diag_3D
from utils import estimate
from utils import buildBasis_list, loglikelihood
from functools import partial


# %% main function
def run_case_study(folder_path):

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

    # %% Estimate the variogram model

    var_model = []
    for y_observed, Xbeta, pt in zip(Y_train_list, Xbeta_train_list, points_train):
        # Get the field
        # field = agg_mean[res].to_numpy()

        field = np.nanmean(y_observed[:, :], axis=1)
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
                                              bin_no=25,
                                              latlon=False, estimator='cressie')

        # fit the variogram with a stable model
        fit_model = gs.Matern(dim=2)
        fit_model.fit_variogram(
            bin_center, gamma, nugget=True, nu=1, anis=False)

        var_model.append(fit_model)

    # % Low rank application
    tEnd = T[0]
    tStart = tEnd - tlag

    results = []
    for lowrank in [1, 0.75, 0.5, 0.25]:

        print("Lowrank:", lowrank, f'Start - [{i}]')

        # estimate the model
        output = estimate(lowrank, Y_train, Xbeta_train, points_train, tStart, tEnd, var_model, hull, points_mesh,
                          Y_test, Xbeta_test, points_test, boundary_polygon)

        # compute the std
        output = compute_unceratinty(output)
        results.append(output)

    # % save the partial resuls (without the hessian evaluation)
    print("Save results")

    # save the results
    filename = folder_path / \
        f'LRSSM_grins_case_study.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(results, f)


# %% Compute Uncertainty function

def compute_unceratinty(model):
    pdim = model['pdim']
    qdim = model['qdim']

    est_covList = model['mesh']
    points_test = model['points_test']
    points_train = model['points_train']
    basis = buildBasis_list(points_test, est_covList)

    y_train = model['y_train']
    Xbeta_train = model['Xbeta_train']
    tlag = y_train.shape[1]

    x_T_hat = model['x_T']
    P_T_hat = model['P_T']
    S11 = model['S11']
    S10 = model['S10']
    S00 = model['S00']
    est_beta = model['beta']
    est_A = model['A']
    est_s2e = model['s2e']
    est_f = model['f']
    est_ks = np.array(model['ks'])
    est_x0 = model['x0']
    est_Sigma0 = model['Sigma0']

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
        y_t=y_train[:, -tlag:],           # ensure JAX array
        Xbeta=Xbeta_train[:, :, -tlag:],
        x_T=x_T_hat,
        P_T=P_T_hat,
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

    # lik_fixed(par)

    ts = time.time()
    hesfun = jax.hessian(lik_fixed)
    IFisher = hesfun(par)
    jax.block_until_ready(IFisher)
    tdelta = time.time() - ts

    SigmaPar = jnp.linalg.inv(IFisher)

    # check if it is postive definte
    chol = jnp.linalg.cholesky(SigmaPar)

    # Compute the std
    std_par = np.sqrt(SigmaPar.diagonal())

    # save the EM full resutls with the uncertainty
    model['Sigma_par'] = SigmaPar
    model['Std_par'] = std_par
    model['stack_par'] = par
    model['stack_dim'] = stack_dim
    model['time_hessian'] = tdelta

    return model

# %% Main section


# main script
if __name__ == "__main__":

    # chek the folder exist
    folder_path = Path(f"./utput/case_study")

    # check that the folder exixst
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print("New folder created successfully!")

    print(f"Run comparison study")
    run_case_study(folder_path)
