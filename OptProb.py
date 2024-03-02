#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: su
"""
import cvxpy as cp
import csv
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

stock = pd.read_csv('Daily_return.csv', header=0)
stock = np.array(stock)
num_day = stock.shape[0]-1
num_stock = stock.shape[1]
return_matrix = np.zeros((num_day, num_stock))
for x in range(1,num_day+1):
    return_matrix[[x-1,]] = stock[[x,]] - stock[[x-1,]]
##covariance matrix and r
Q = np.cov(return_matrix.T)


kk = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#kk = [i*10 for i in kk]
for k in kk: 


    Q_new = pd.read_csv('Correlation_matrix.csv',header=0)
    Q_new = np.array(Q_new)
    r = np.array(return_matrix).mean(axis=0)
    new_col = np.zeros((num_stock,1))
    new_row = np.zeros((num_stock+1,1)).T
    #apply cholesky decomposition on Q
    #L = np.array(fractional_matrix_power(Q_new,0.5),dtype = float)
    L = pd.read_csv('Correlation_matrix_CD.csv',header=0)
    L = np.array(L, dtype = float)
    L = np.append(L, new_col, 1)
    L = np.append(L, new_row, 0)  #new covariance matrix for further calculation
    r = np.append(r, -1)


    m = 201
    n = 201
    p = 1
    n_i = 201
    f = np.array(np.append(np.repeat(0,200),-1),dtype = float)
    A = [np.array(k*L, dtype = float)]
    b = [np.repeat(0,201)]
    c = [r]
    d = [0]
    x0 = np.random.randn(n)
    for i in range(m-1):
        # temp = np.repeat(np.array([0., 1., 0.])[None, :], 1, axis=0, dtype = float)
        temp = np.array([np.repeat(0,201)], dtype = float)
        A.append(temp)
        temp = np.array(0, dtype = float)
        b.append(temp)
        temp = np.array(np.repeat(0,201), dtype = float)
        temp[[i]] = 1 
        temp = np.array(temp, dtype = float)
        c.append(temp)
        temp = np.array(0, dtype = float)
        d.append(0)
    
    F = np.array([np.repeat(1,201)])
    F[:,200]=0
    F = np.array(F, dtype = float)
    #F = np.array(F, dtype = float)
    g = np.array(1, dtype = float)
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T @ x),
                      soc_constraints + [F @ x == g])
    prob.solve()
    
    tol = 10e-6
    x_index = [i for i,j in enumerate(x.value > tol) if j] #the selected stock
    
    
    
    # Print result.
    print("The optimal value when k=" + str(k) + " is", prob.value)
    print("The selected stock when k=" + str(k) + " is number:", x_index)

################################

kk = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

kk = [0.001,0.002,0.003,0.004]
optim_value = []
#kk = [i*10 for i in kk]
for k in kk: 

    r = np.array(return_matrix).mean(axis=0)
    L = pd.read_csv('Correlation_matrix_CD.csv',header=0)
    L = np.array(L, dtype = float).T

    # A = np.array(k*L, dtype = float)
    # c = r
 #   x0 = np.random.randn(n)
    # Define and solve the CVXPY problem.
    x = cp.Variable(200)
    t = cp.Variable(1)
    objective = cp.Maximize(t)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [sum(x) ==1, x>=0, cp.SOC(r @ x -t , k*L @ x) ]
    prob = cp.Problem(objective, soc_constraints)
    prob.solve()
    
    optim_value.append(prob.solve())
    
    tol = 0.03
    x_index = [i for i,j in enumerate(x.value > tol) if j] #the selected stock
    
    
    
    # Print result.
    print("The optimal value when k=" + str(k) + " is", prob.value)
    print("The selected stock when k=" + str(k) + " is number:", x_index)


plt.plot(kk, optim_value)
# r.tofile('r.csv',sep=',',format='%10.5f')
# L.tofile('sqrt_Q.csv',sep=',',format='%10.5f')


########### part c

delta = [3800, 3900, 4000, 4200, 4400]
optim_value_ = []
#kk = [i*10 for i in kk]
for k in delta: 

    Q_new = pd.read_csv('Correlation_matrix.csv',header=0)
    Q_new = np.array(Q_new)
    r = np.array(return_matrix).mean(axis=0)
    L = pd.read_csv('Correlation_matrix_CD.csv',header=0)
    L = np.array(L, dtype = float)


    m = 200
    n = 200
    p = 1
    n_i = 200
    f = r
    A = [L]  ##upeer triangular matrix
    c = [np.array(np.repeat(0,m), dtype = float)]
    d = [np.array(np.sqrt(2*k), dtype = float)]
    x0 = np.random.randn(n)
    
    for i in range(m):
        # temp = np.repeat(np.array([0., 1., 0.])[None, :], 1, axis=0, dtype = float)
        temp = np.array([np.repeat(0,200)], dtype = float)
        A.append(temp)
        temp = np.array(np.repeat(0,200), dtype = float)
        temp[[i]] = 1 
        temp = np.array(temp, dtype = float)
        c.append(temp)
        temp = np.array(0, dtype = float)
        d.append(0)

    
    F = np.array([np.repeat(1,200)], dtype = float)
    g = np.array(1, dtype = float)
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i] , A[i] @ x) for i in range(m+1)
    ]
    prob = cp.Problem(cp.Maximize(f.T @ x),
                      soc_constraints + [F @ x == g])
    prob.solve()
    
    optim_value_.append(prob.solve())

    
    tol = 10e-6
    x_index = [i for i,j in enumerate(x.value > tol) if j] #the selected stock
    
    
    
    # Print result.
    print("The optimal value when delta=" + str(k) + " is", prob.value)
    print("The selected stock when delta=" + str(k) + " is number:", x_index)
    
plt.plot(delta, optim_value_)

