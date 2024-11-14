"""
Second-order instantaneous causal analysis

"""
import numpy as np
import matplotlib.pyplot as plt
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


#==Simulation with aritifical data=============================================
#Main parameters
n = 4 #dimension of data (length of single time series)
scenarios = 6 #how many scenarios (e.g. sample sizes)
N_obsvalues = np.round(np.power(10,1 + 0.5*np.linspace(1,scenarios,scenarios))).astype(int) #Sample sizes for each scenario
maxiter = 1000; #how many iterations per scenario

#For storing results:
correctdecisions = np.zeros([scenarios,maxiter])
correctpercentage = np.zeros(scenarios)

#LOOP over scenarios
for scenario in range(scenarios):
    #Choose number of data points  for this scenario
    N_obs = N_obsvalues[scenario]
    
    #LOOP over repetitions (observations of data)
    for iter in range(maxiter):
        ######################
        #Create simulated data
        ######################
        
        #CHOOSE PARAMETERS defining statistics of data
        #causal regression coeff
        alphatrue = 0.2 # 0.5*rand()+.2;
        #std of noise in regression equation
        beta = 0.5
        
        #Create r1, autoregressive coeff for real data (regressor)
        r1 = 0.3 #0.5*rand()+.2;
        #Create r2, autoregressive coeff for noise, zero by default, but does not have to be
        r2 = 0.0
        
        #CREATE DATA
        #Initialize X and residual noise
        X = np.zeros([n,N_obs])
        E = np.zeros([n,N_obs])
        #create starting point for each time series
        X[0,:] = np.random.randn(1,N_obs)
        E[0,:] = np.random.randn(1,N_obs)
        #create autoregressive data, potentially for N as well
        for t in range(1,n):
            X[t,:] = r1*X[t-1,:] + np.random.randn(1,N_obs)
            E[t,:] = r2*E[t-1,:] + np.random.randn(1,N_obs)
        #standardize X,N
        for t in range(n):
            X[t,:] = X[t,:]/np.std(X[t,:])
            E[t,:] = E[t,:]/np.std(E[t,:])
            
        #Create Y by regression equation
        Y = alphatrue*X + beta*E
        
        #DEBUGGING: change ordering to check that the estimated ordering changes as well
        # tmp=X; X=Y; Y=tmp
        
        ###############################
        #   CALL ESTIMATION FUNCTION
        ###############################
        
        #This gives a scalar whose sign gives direction, and abs value some kind of certainty (?)
        decisionvariable = soc_est(X,Y)
        
        #ANALYTICS
        
        #Store result for later analysis
        correctdecisions[scenario,iter] += int(decisionvariable<0)
    # end of iter
    correctpercentage[scenario] = np.sum(correctdecisions[scenario,:])/maxiter*100
    print(('N_obs = %d, correct: %3.2f,') % (N_obs, correctpercentage[scenario]))
# end


        
plt.plot(np.log10(N_obsvalues), correctpercentage)

plt.title("Causal discovery success")
plt.ylabel("Percentage correct")
plt.xlabel("log10 sample sizes")