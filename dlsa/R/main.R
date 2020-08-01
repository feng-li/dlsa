
## This file is the main simulation file

# please set the directory to be the current file
source("wlse_func.R")
source("dlsa_alasso_func.R")
source("func_others.R")
source("simulator.R")
source("comparison/compare_func.R")

# sample size
N = 1000

# true parameter setting
suppressMessages({
  options(warn=-1)
  args = commandArgs(trailingOnly = TRUE)
  cat("PID: ", Sys.getpid(), "\n")

  cat(args, "\n")
  case = as.numeric(args)
  
if (case == 1)
{
  reg_type = "linear"
  beta = c(3, 0, 0, 1.5, 0, 0, 2, 0)
}

if (case == 2)
{
  reg_type = "logistic"
  beta = c(3, 0, 0, 1.5, 0, 0, 2, 0) 
}

if (case == 3)
{
  reg_type = "poisson"
  beta = c(0.8,0,0,1,0,0,-0.4,0)
}

if (case == 4)
{
  reg_type = "cox"
  beta = c(0.8,0,0,1,0,0,0.6,0)
}

if (case == 5)
{
  reg_type = "ordered_probit"
  beta = c(0.8,0,0,1,0,0,0.6,0)
}



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
Nrep = 500
Ns = c(10000,20000,100000)
Ks = c(5,5,10)
p = length(beta)
non0_ind = which(beta!=0)
n_non0 = length(non0_ind)

# store the results
theta_global = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_wlse = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_oneshot = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_jordan_onestep = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_jordan = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_onestep = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_bic = rep(list(matrix(0, nrow = p, ncol = Nrep)), length(Ns))
theta_oracle = rep(list(matrix(0, nrow = n_non0, ncol = Nrep)), length(Ns))

ms = matrix(0, nrow = 2, length(Ns))
cm = matrix(0, nrow = 2, length(Ns))
iid = c(T, F)
res = rep(list(list(theta_global = theta_global,
               theta_wlse = theta_wlse,
               theta_oneshot = theta_oneshot,
               theta_jordan_onestep = theta_jordan_onestep,
               theta_jordan = theta_jordan,
               theta_onestep = theta_onestep,
               theta_bic = theta_bic,
               theta_oracle = theta_oracle)), 2)

for (s in 1:2)
{
  for (i in 1:length(Ns))
  {
    N = Ns[i]
    K = Ks[i]
    for (r in 1:Nrep)
    {
      cat(N, "\n")
      cat(r, " | ")
      set.seed(r)
      
      # simulate X and Y
      X = simu.X(N, p, K = K, iid = iid[s])
      Y = simu.Y(X, beta, reg_type)
      
      # global estimator
      global_est = reg(X, Y, reg_type)
      
      # WLSE estimator
      beta_est = dlsa(X, Y, K = K, reg_type)
      
      # CSL onestep estimator
      beta_jordan_onestep = Jordan.distribute(X, Y, K, reg_type, iter.max = 1, init_oneshot = F)
      
      # oracle estimator
      oracle_est = reg(X[, non0_ind], Y, reg_type)
      
      # Adaptive Lasso estimator
      lsa_lars = lsa.distribute(theta_mat = beta_est$theta_mat, 
                                Sig_inv_list = beta_est$Sig_inv_list, 
                                intercept = F,
                                n = nrow(X))
      
      res[[s]]$theta_bic[[i]][,r] = lsa_lars$beta.bic
      res[[s]]$theta_oracle[[i]][,r] = oracle_est$theta
      res[[s]]$theta_global[[i]][,r] = global_est$theta
      res[[s]]$theta_wlse[[i]][,r] = beta_est$theta
      res[[s]]$theta_jordan_onestep[[i]][,r] = beta_jordan_onestep$theta
      res[[s]]$theta_oneshot[[i]][,r] = beta_est$theta_onehot
      
    }
    
    # Estimation bias
    bias_global = res[[s]]$theta_global[[i]] - beta
    bias_oneshot = res[[s]]$theta_oneshot[[i]] - beta
    bias_wlse = res[[s]]$theta_wlse[[i]] - beta
    bias_jordan_onestep = res[[s]]$theta_jordan_onestep[[i]] - beta
    
    # estimated model size
    ms[s, i] = mean(colSums(res[[s]]$theta_bic[[i]]!=0))
    
    # percentage of correct models identified
    cm[s, i] = mean(apply(res[[s]]$theta_bic[[i]]!=0, 2, function(x) x==(beta!=0)))
    
    
    # output the result
    cat("N = ", N, " K = ", K, "\n",
        "RMSE: \n",
        "Oneshot:         ", rowRMSE.K(bias_oneshot), "\n",
        "WLSE:            ", rowRMSE.K(bias_wlse), "\n",
        "Jordan Onestep:  ", rowRMSE.K(bias_jordan_onestep), "\n\n",
        "RME: \n",
        "Oneshot:         ", RMSE.ratio.K(bias_global, bias_oneshot), "\n",
        "WLSE:            ", RMSE.ratio.K(bias_global, bias_wlse), "\n",
        "Jordan Onestep:  ", RMSE.ratio.K(bias_global, bias_jordan_onestep), "\n\n"
    )
  }
}

save(res, file = paste0("../data/simu/case", case,  ".rda"))
})
### output tex tables

model_names = c("OS", "CSL", "DLSA", "SDLSA")
settings = c("1: i.i.d Covariates", "2: Heterogenous Covariates")
ms1 = specify_decimal(ms, 2)
cm1 =  specify_decimal(cm, 2)


for (s in 1:2)
{
  cat("\\multicolumn{13}{c}{\\sc Setting", settings[s], "} \\\\ \n")
  for (i in 1:length(Ns))
  {
    N = Ns[i]
    K = Ks[i]
    # Estimation bias
    bias_global = res[[s]]$theta_global[[i]] - beta
    bias_oneshot = res[[s]]$theta_oneshot[[i]] - beta
    bias_wlse = res[[s]]$theta_wlse[[i]] - beta
    bias_jordan_onestep = res[[s]]$theta_jordan_onestep[[i]] - beta
    bias_oracle =  res[[s]]$theta_oracle[[i]] - beta[non0_ind]
    bias_bic = res[[s]]$theta_bic[[i]] - beta
    
    rs = rep("", 4)
    rs[1] = paste(RMSE.ratio.K(bias_global, bias_oneshot), collapse = " & ")
    rs[2] = paste(RMSE.ratio.K(bias_global, bias_jordan_onestep), collapse = " & ")
    rs[3] = paste(RMSE.ratio.K(bias_global, bias_wlse), collapse = " & ")
    
    tmp = rep("-", p)
    #tmp[non0_ind] = RMSE.ratio.K(bias_oracle, bias_bic[non0_ind,])
    tmp[non0_ind] = RMSE.ratio.K(bias_global[non0_ind,], bias_bic[non0_ind,])
    rs[4] = paste(tmp, collapse = " & ")
    
    
    for (r in 1:4)
    {
      if (r == 1){
        # if (i==3)
        #   N = "100000"
        cat(N/1000, " & ", K, " & ")
      }else{
        cat( " &  & ")
      }
      cat(model_names[r], " & ", rs[r], " & ")
      if (r == 4)
        cat(ms1[s, i], " & ", cm1[s, i], "\\\\ \n")
      else{
        cat(" & \\\\ \n")
      }
    }
    if (i == 3)
      cat("\\hline  \n")
    else
      cat(rep("&", p+4), "\\\\  [-1.1em]\n")
  }
}



