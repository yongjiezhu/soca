"""
Second-Order instantaneous Causal Analysis

"""
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.covariance import GraphicalLasso



def soc_est(X,Y):
    """Estimate Second-order Causal model method in Our paper
    Args:
        X: size as (t, N_obs)
        Y: size as (t, N_obs); t is length of each time series (at least 2)
           N_obs is the number of observations.
        
    Return:
        a scalar value, differenve of mutual informations of residuals,
        if it is negative, then direction x->y detected.
    """
    # Standardize data for each time point (=row of X or Y)
    X -= np.mean(X, axis=1)[:, None]
    X = X / np.std(X, axis=1)[:, None]
    
    Y -= np.mean(Y, axis=1)[:, None]
    Y = Y / np.std(Y, axis=1)[:, None]
    
    #Compute covariances and their determinants for data variables
    Cx = np.cov(X)
    det_x = np.linalg.det(Cx)
    
    Cy = np.cov(Y)
    det_y = np.linalg.det(Cy)
    
    ### Compute covariances and determinants of residuals e_xy, e_yx
    
    #Estimate regression coefficient (assuming X and Y standardized)
    alphaest = np.cov(X.T.flatten(),Y.T.flatten())[0,1]
    
    #Residual for x->y direction of regression
    E_xy = Y - alphaest*X
    Ce_xy = np.cov(E_xy)
    det_e_xy = np.linalg.det(Ce_xy)
    
    #Residual for y->x direction of regression
    E_yx = X - alphaest*Y
    Ce_yx = np.cov(E_yx)
    det_e_yx = np.linalg.det(Ce_yx)
    
    #Compute covariance and residual of concatenated data########
    #x->y direction of regression
    Z_xy = np.concatenate((X,E_xy), axis=0)
    Cz_xy = np.cov(Z_xy)
    det_z_xy = np.linalg.det(Cz_xy)
    #y->x direction of regression
    Z_yx = np.concatenate((Y,E_yx), axis=0)
    Cz_yx = np.cov(Z_yx)
    det_z_yx = np.linalg.det(Cz_yx)
    
    #Compute mutual infos in both directions
    mutuinfo_xy = np.log(det_x) + np.log(det_e_xy) - np.log(det_z_xy) 
    mutuinfo_yx = np.log(det_y) + np.log(det_e_yx) - np.log(det_z_yx)
    
    return mutuinfo_xy-mutuinfo_yx