"""
soca Simulation with aritifical data

"""
import numpy as np
import matplotlib.pyplot as plt
from soca import soc_est
#from sklearn.covariance import GraphicalLasso


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