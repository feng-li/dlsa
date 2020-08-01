
## This file is to compare the dlsa method with the CSL method proposed by
## Michael I. Jordan, Jason D. Lee & Yun Yang (JASA, 2019)

source("../wlse_func.R")
source("../dlsa_alasso_func.R")
source("../func_others.R")
source("../simulator.R")
source("compare_func.R")

# sample size
N = 1000

# true parameter setting
reg_type = "logistic"
beta = c(3, 0, 0, 1.5, 0, 0, 2, 0, 0) 

# reg_type = "cox"
# beta = c(0.8,0,0,1,0,0,0.6,0)
# # 
reg_type = "poisson"
beta = c(0.8,0,0,1,0,0,-0.4,0)
# 
# reg_type = "linear"
# beta = c(3, 0, 0, 1.5, 0, 0, 2, 0, 0) 

# reg_type = "ordered_probit"
# beta = c(0.8,0,0,1,0,0,0.6,0)


p = length(beta)


# data generation
set.seed(1234)
X = simu.X(N, p, K = 5)
Y = simu.Y(X, beta, reg_type)

# Demo: DLSA method and CSL method
beta_global = reg(X, Y, reg_type)

beta_est = dlsa(X, Y, K = 5, reg_type)
beta_jordan = Jordan.distribute(X, Y, K = 5, iter.max = 1, reg_type)

# global_hess = solve(vcov(beta_global))/N
# our_hess = beta_est$Sig_inv/N
# jordan_hess = beta_jordan$hess

########################################################################
### Simulation repeat Nrep times
Nrep = 200
Ns = c(500, 1000)*20
K = 5
p = length(beta)

# store the results
theta_global = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_wlse = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_oneshot = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_jordan_onestep = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_jordan = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_onestep = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))


for (i in 1:length(Ns))
{
  N = Ns[i]
  for (r in 1:Nrep)
  {
    cat(N, r, "\r")
    set.seed(r)
    
    # simulate X and Y
    X = simu.X(N, p, K = K)
    Y = simu.Y(X, beta, reg_type)
    
    # global estimator
    global_est = reg(X, Y, reg_type)
    
    # WLSE estimator
    beta_est = dlsa(X, Y, K = K, reg_type)
    
    # CSL onestep estimator
    beta_jordan_onestep = Jordan.distribute(X, Y, K, reg_type, iter.max = 1, init_oneshot = F)
    
    # CSL iterative estimator
    #beta_jordan = logistic.Jordan.distribute(X, Y, K, reg_type, iter.max = 80, init_oneshot = F)
    
    # Global onestep estimator (use local estimator as initial)
    # beta_global_onestep = logistic.global.onestep(X, Y, K, iter.max = 1, 
    #                                               init_est = "local", local.hessian = F)
    
    theta_global[[i]][,r] = global_est$theta
    theta_wlse[[i]][,r] = beta_est$theta
    theta_jordan_onestep[[i]][,r] = beta_jordan_onestep$theta
    #theta_jordan[[i]][,r] = beta_jordan$theta
    theta_oneshot[[i]][,r] = beta_est$theta_onehot
    #theta_onestep[[i]][,r] = beta_global_onestep
  }
  
  # Estimation bias
  bias_global = theta_global[[i]] - beta
  bias_oneshot = theta_oneshot[[i]] - beta
  bias_wlse = theta_wlse[[i]] - beta
  bias_jordan_onestep = theta_jordan_onestep[[i]] - beta
  bias_jordan = theta_jordan[[i]] - beta
  #bias_onestep = theta_onestep[[i]] - beta
  
  
  # output the result
  cat("N = ", N, " K = ", K, "\n",
      "RMSE: \n",
      "Oneshot:         ", rowRMSE.K(bias_oneshot), "\n",
      "WLSE:            ", rowRMSE.K(bias_wlse), "\n",
      #"Global Onestep:  ", rowRMSE.K(bias_onestep), "\n",
      "Jordan Onestep:  ", rowRMSE.K(bias_jordan_onestep), "\n\n",
      #"Jordan Iterative:", rowRMSE.K(bias_jordan), "\n \n",
      "RME: \n",
      "Oneshot:         ", RMSE.ratio.K(bias_global, bias_oneshot), "\n",
      "WLSE:            ", RMSE.ratio.K(bias_global, bias_wlse), "\n",
      #"Global Onestep:  ", MSE.ratio.K(bias_global, bias_onestep), "\n",
      "Jordan Onestep:  ", RMSE.ratio.K(bias_global, bias_jordan_onestep), "\n\n",
      #"Jordan Iterative:", MSE.ratio.K(bias_global, bias_jordan), "\n\n"
       "Jordan Bias: ", rowMeans.K(bias_jordan_onestep), "\n",
       "Jordan SD  : ", rowSd.K(bias_jordan_onestep), "\n\n"
      # "WLSE Bias:   ", rowMeans.K(bias_wlse), "\n",
      # "WLSE SD  :   ", rowSd.K(bias_wlse), "\n\n"
      )
}





