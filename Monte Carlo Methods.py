#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install sobol_seq')
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skewnorm
from scipy.stats import norm
import sobol_seq
from typing import Iterable, Any
import time
import seaborn as sns
sns.set()

def cir_st_disc(s_0, alpha, b, sigma, k: int= 20, delta: float=1, T= 1, seed= None):
    """  Simulate one path of CIR Model  """
    np.random.seed(seed)
    random.seed(seed)

    # Instance stock price ts and append it first value
    s_t = []
    s_t_neg= []
    s_t.append(s_0)
    s_t_neg.append(s_0)
    
    k= int(T/delta)
    
    for i in range(k):
        s_t_value= np.maximum(s_t[-1], 0) # because a price can't be negative, so we enforce our processus
        # to be positive or 0

        # We generata a normal number
        # cuz' (W_{t+1} - W_{t} follow a N(0, delta)
        epsilon= np.random.normal()

        mu= alpha*(b - s_t_value)*delta
        sigma_s_t= sigma * np.sqrt(s_t_value * delta)

        d_s_t= mu + sigma_s_t*epsilon
        d_s_t_neg= mu - sigma_s_t*epsilon

        s_t.append(s_t_value + d_s_t)
        s_t_neg.append(s_t_value + d_s_t_neg)

    return np.array(s_t), np.array(s_t_neg)


# In[ ]:


def simulation_callvalue(num_exec, n, s_0, alpha, b, sigma, k, strike, r, T
                         , delta):
    start_time= time.time()
    original_value= []
    antithetic_value= []
    
    for t in range(num_exec): # Simulate num_exec number of CALL value
        stock_s_t= []
        stock_s_t_neg= []
        for i in range(n): # simulate n path to compute the call value (estimate expectation with mean)
            s_t, s_t_neg= cir_st_disc(s_0=s_0, alpha=alpha, b=b, sigma=sigma, k=k
                                    , delta= delta)
            stock_s_t.append(s_t)
            stock_s_t_neg.append(s_t_neg)

        # Cash flow computing
        opt_cachflow= np.array([execution_opt_diff(s_t= s, strike= strike) for s in stock_s_t])
        opt_cachflow_neg= np.array([execution_opt_diff(s_t= s, strike= strike) for s in stock_s_t_neg])

        # Cash flow actualisation
        actu_opt_cachflow= np.array([actualization_execution_opt(c, r= r, T= T) for c in opt_cachflow])
        actu_opt_cachflow_neg= np.array([actualization_execution_opt(c, r= r, T= T) for c in opt_cachflow_neg])

        # Compute CALL value
        original_value.append(np.mean(actu_opt_cachflow, axis= 0))
        antithetic_value.append(np.mean((actu_opt_cachflow + actu_opt_cachflow_neg)/2, axis= 0))

    print("Execution %s seconds" % (time.time()-start_time))
    
    return original_value, antithetic_value


# In[ ]:


def compute_corr_coef(n, s_0, alpha, b, sigma, k, strike, r, T, delta, num= 100):
    
    start_time= time.time()
    
    stock_s_t= []
    for t in range(num):
        for i in range(n): # simulate n path to compute the call value (estimate expectation with mean)
            s_t, _= cir_st_disc(s_0=s_0, alpha=alpha, b=b, sigma=sigma, k=k
                                    , delta= delta)
            stock_s_t.append(s_t)
            
        opt_cachflow= np.array([execution_opt_diff(s_t= s, strike= strike) for s in stock_s_t])
        opt_cachflow_geom= np.array([execution_geom_diff(s_t= s, strike= strike) for s in stock_s_t])

    # Link between mean and geom
    # No actualisation because it change nothing to do it for both for the cov/var
    opt_cachflow_geom= opt_cachflow_geom - np.mean(opt_cachflow_geom)
    var_cov_mat= np.cov([opt_cachflow_geom, opt_cachflow])
    beta= var_cov_mat[0, 1] / var_cov_mat[0, 0]
    
    print("Execution %s seconds" % (time.time()-start_time))  
    
    return beta


def simulation_control(num_exec, n, s_0, alpha, b, sigma, k, strike, r, T, delta, beta= None):
    
    start_time= time.time()
    
    if beta is None:
        beta= 1 
        
    original_value= []
    control_value= []
    
    for t in range(num_exec): # Simulate num_exec number of CALL value
        stock_s_t= []
        for i in range(n): # simulate n path to compute the call value (estimate expectation with mean)
            s_t, _= cir_st_disc(s_0=s_0, alpha=alpha, b=b, sigma=sigma, k=k
                                    , delta= delta)
            stock_s_t.append(s_t)

        # Cash flow computing
        #On sait que la moyenne arithmétique est un majorant de la moyenne géométrique. 
        #Cela peut donc laisser penser qu'il existe une corrélation non nulle entre les deux.
        opt_cachflow= np.array([execution_opt_diff(s_t= s, strike= strike) for s in stock_s_t])
        opt_cachflow_geom= np.array([execution_geom_diff(s_t= s, strike= strike) for s in stock_s_t])
        
        # Compute mean of both
        mean_cashflow= (1/2)*(opt_cachflow + beta*opt_cachflow_geom)
        
        # Cash flow actualisation
        actu_opt_cachflow= np.array([actualization_execution_opt(c, r= r, T= T) 
                                     for c in opt_cachflow])
        actu_opt_cachflow_control= np.array([actualization_execution_opt(c, r= r, T= T) 
                                     for c in mean_cashflow])
        
        # Compute CALL value
        control_value.append(np.mean(actu_opt_cachflow_control, axis= 0))
        original_value.append(np.mean(actu_opt_cachflow, axis= 0))
        
    print("Execution %s seconds" % (time.time()-start_time))
    
    return original_value, control_value


# In[ ]:


def compute_best_n_l(variance_serie, h_l_serie, l, rms_accuracy):
    assert variance_serie.shape == h_l_serie.shape
    summed_value= np.sum(np.sqrt(variance_serie/h_l_serie))
    rooted= np.sqrt(variance_serie[l]*h_l_serie[l])
    return int(summed_value * rooted * (2/(rms_accuracy**2)))

def mlmc_mean(serie, h_l):
    # the paper ignore sO
    return np.sum(0.5*(serie[1:] + serie[:-1])*h_l)
#     return np.mean(serie)

def compute_L_0_step(n_0, h_L, s_0, alpha, b, sigma, k, delta, strike):
    actualized_cashflow_0= []
    for i in range(n_0):
        s_l, _= cir_st_disc(s_0=s_0, alpha=alpha, b=b, sigma=sigma, k=k
                             , delta= h_L)
        # Compute mean and activation value
        execution_value= mlmc_mean(s_l, h_L)
        # Actualisation
        actualized_cashflow_0.append(actualization_execution_opt(execution_value - strike, r, T))
        
    return actualized_cashflow_0

def compute_l_step(n_l, l, s_0, alpha, b, sigma, k, strike, seed_key= 55):
    np.random.seed(seed_key)
    
    assert l > 0
    
    h_l= T/((M**l)*2)
    h_l1= T/((M**(l-1))*2)
    
    actualized_cashflow= []
    actualized_cashflow_1= []
    o= []
    for i in range(n_l):
        path_seed= int(np.random.uniform(0, pow(2, 32))) # we control each path to be the same
        # Only the timestemp change but the same path is simulated
        s_l, _= cir_st_disc(s_0=s_0, alpha=alpha, b=b, sigma=sigma, k=k
                             , delta= h_l, seed= path_seed)
        s_l1, _= cir_st_disc(s_0=s_0, alpha=alpha, b=b, sigma=sigma, k=k
                             , delta= h_l1, seed= path_seed)

        # Compute mean and activation value
        execution_value= mlmc_mean(s_l, h_l)
        execution_value_1= mlmc_mean(s_l1, h_l1)
        # Actualisation
        actualized_cashflow.append(actualization_execution_opt(execution_value - strike, r, T))
        actualized_cashflow_1.append(actualization_execution_opt(execution_value_1 - strike, r, T))
        
        o.append(s_l)

    return actualized_cashflow, actualized_cashflow_1, o

def convergence_test(Y_L, M, Y_L_1, rms_accuracy):
    left_side= np.abs(Y_L - (Y_L_1/M))
    right_side= (rms_accuracy*(M**2 - 1))/np.sqrt(2)
    return left_side < right_side

def looking_for_best_nl(control_seed, n_0, s_0, alpha, b, sigma, k, strike
                        , rms_accuracy= 0.001):
    # first step L= 0
    start_time= time.time()

    L= 0 # INitialisation
    #n_0= 10**4 # From article
    M= 2 # From class
    h_L= T/((M**L)*2)
#     rms_accuracy= 0.001 # User definition

    np.random.seed(control_seed)

    v_L= []
    h_L_stock= []
    n_L_stock= []

    # Compute cash flos for L= 0
    actualized_cashflow_0= compute_L_0_step(n_0, h_L, s_0, alpha, b, sigma, k, h_L, strike)

    v_0= np.std(actualized_cashflow_0)**2

    v_L_copy= v_L.copy()
    v_L_copy.append(v_0)

    h_L_stock_copy= h_L_stock.copy()
    h_L_stock_copy.append(h_L)

    best_n_0= compute_best_n_l(np.array(v_L_copy), np.array(h_L_stock_copy)
                     , 0, rms_accuracy)

    # save best N_O for the moment
    n_L_stock.append(best_n_0)
    test_value= False
    while (L < 2) and (not test_value):
        print(L)
        L += 1
        # Compute values for l = 0
        # Compute cash flow for L= 0
        actualized_cashflow_0= compute_L_0_step(n_L_stock[0], h_L, s_0, alpha, b, sigma, k, h_L, strike)
        v_0= np.std(actualized_cashflow_0)**2

        # Initialize to this value (for the uncomputed n_l)
        n_L_stock.append(n_0)

        v_l_diff_stock= []
        v_l_diff_stock.append(v_0) # add to compute best n_l later

        for i in range(L): # compute values for l > 0 | l <= L
            l= i+1

            seed_key= int(np.random.uniform(0, pow(2, 32)))
            actualized_cashflow, actualized_cashflow_1, okok= compute_l_step(n_L_stock[l], l, s_0, alpha, b
                                                                       , sigma, k, strike= strike
                                                                       , seed_key= seed_key)

            diff_cash_flow= np.array(actualized_cashflow) - np.array(actualized_cashflow_1)
    #         diff_cash_flow_stock.append(diff_cash_flow)
            v_l_diff_stock.append(np.std(diff_cash_flow)**2)

        h_L_stock_copy= [h_L] + [T/((M**(j+1))*2) for j in range(L)]
        # compute best n_l l \in 0 ... L ==> (n_O, n_1 ... n_L)
        n_L_stock= [compute_best_n_l(np.array(v_l_diff_stock), np.array(h_L_stock_copy)
                , j, rms_accuracy) for j in range(L+1)]

        # We got best n_l for all l \in O, .. , L, now compute the good CF diff
        # Stock diff of CF
        actualized_cashflow_0= compute_L_0_step(n_L_stock[0], h_L, s_0, alpha, b, sigma, k, h_L, strike)
        diff_cash_flow_stock= []
        diff_cash_flow_stock.append(np.array(actualized_cashflow_0))
        for i in range(L): # compute values for l > 0 | l <= L
            l= i+1

            seed_key= int(np.random.uniform(0, pow(2, 32)))
            actualized_cashflow, actualized_cashflow_1, okok= compute_l_step(n_L_stock[l], l, s_0, alpha, b
                                                                       , sigma, k, strike= strike
                                                                       , seed_key= seed_key)

            diff_cash_flow= np.array(actualized_cashflow) - np.array(actualized_cashflow_1)
            diff_cash_flow_stock.append(diff_cash_flow)
            v_l_diff_stock.append(np.std(diff_cash_flow)**2)

        # Convergence test if L >= 2, can stop the process
        if L >= 2:
            y_l= np.sum(diff_cash_flow_stock[-1]/n_L_stock[-1])
            y_l1= np.sum(diff_cash_flow_stock[-2]/n_L_stock[-2])
            test_value= convergence_test(y_l, M, y_l1, rms_accuracy)

    print("Execution %s seconds" % (time.time()-start_time))
    
    return n_L_stock, diff_cash_flow_stock


# In[ ]:


def call_value_mlmc(diff_cash_flow_stock, n_L_stock):
    diff_value= np.array([np.sum(d) for d in diff_cash_flow_stock])/np.array(n_L_stock)
    return np.sum(diff_value)

def compute_call_value_mlmc(control_seed, n_L_stock, s_0, alpha, b, sigma, k, h_L, strike):
    np.random.seed(control_seed)
    
#     start_time= time.time()

    L= len(n_L_stock) # true L is that value minus 1, because l \in 0, .. , L
    h_l= [T/((M**(j))*2) for j in range(L)]

    v_l_diff_stock= []
    #Stock diff of CF
    actualized_cashflow_0= compute_L_0_step(n_L_stock[0], h_l[0], s_0, alpha, b, sigma, k, h_L, strike)
    diff_cash_flow_stock= []
    diff_cash_flow_stock.append(np.array(actualized_cashflow_0))
    actualized_cashflow_stock= []
    actualized_cashflow_stock.append(actualized_cashflow_0)
    for i in range(L-1): 
        l= i+1
        seed_key= int(np.random.uniform(0, pow(2, 32)))
        actualized_cashflow, actualized_cashflow_1, okok= compute_l_step(n_L_stock[l], l, s_0, alpha, b
                                                                   , sigma, k, strike= strike
                                                                   , seed_key= seed_key)
        actualized_cashflow_stock.append(actualized_cashflow)
        diff_cash_flow= np.array(actualized_cashflow) - np.array(actualized_cashflow_1)
        diff_cash_flow_stock.append(diff_cash_flow)
        v_l_diff_stock.append(np.std(diff_cash_flow)**2)

    call_val= call_value_mlmc(diff_cash_flow_stock, n_L_stock)
#     print("Execution %s seconds" % (time.time()-start_time))
    
    return call_val, v_l_diff_stock, diff_cash_flow_stock, actualized_cashflow_stock


# In[ ]:


def cir_st_lowdiscr(s_0, alpha, b, sigma, p, k: int= 20, delta: float=1):

    # Instance stock price ts and append it first value
    s_t = []
    s_t.append(s_0)
    sob = sobol_seq.i4_sobol_generate(k, p)
    for i in range(k):
        s_t_value= np.maximum(s_t[-1], 0) 
        mu= alpha*(b - s_t_value)
        sigma_s_t= sigma * np.sqrt(s_t_value * delta) 
        d_s_t= delta*mu + sigma_s_t*norm.ppf(sob[p-1][i])
        s_t.append(s_t_value + d_s_t)
    return np.array(s_t)

def lowdiscr_path(s_0, alpha, b, sigma,n, k: int= 20, delta: float=1):
    path = []
    for p in range(1,n+1):
        path.append(cir_st_lowdiscr(s_0, alpha, b, sigma, p, k, delta))
    return path

