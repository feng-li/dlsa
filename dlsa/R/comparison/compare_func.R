# calculate the gradient and hessian matrix for logistic regression given theta
logistic.grad.hess<-function(X, Y, beta, Hessian = F)
{
  Xbeta = X%*%beta
  prob01 = 1/as.vector(1+exp(Xbeta))*cbind(1, exp(Xbeta))
  
  # obtain gradient and Hessian matrix
  prob = as.vector(exp(Xbeta)/(1+exp(Xbeta)))
  grad = colSums((Y - prob)*X)
  
  # if Hessian = F, return gradient 
  if (!Hessian)
    return(grad)
  # else then return hessian matrix and gradient
  hess = t(X)%*%(prob*(1-prob)*X)
  return(list(grad = grad, hess = hess))
}


# calculate the gradient and hessian matrix for Cox given theta
# reference: https://en.wikipedia.org/wiki/Proportional_hazards_model
cox.grad.hess<-function(X, Y, beta, Hessian = F)
{
  p = ncol(X)
  exp_Xbeta = exp(X%*%beta)
  
  timeY = Y[,'time']
  statusY = Y[,'status']
  indY1 = which(statusY == 1)
  
  timeY1 = timeY[indY1]
  time_mat = outer(timeY, timeY1, ">=")*1
  
  exp_Xbeta_mat = as.vector(exp_Xbeta)*time_mat
  W = t((1/colSums(exp_Xbeta_mat))*t(exp_Xbeta_mat))
  X_j = t(W)%*%X
  X_i = X[indY1,]
  grad = colSums(X_i - X_j)
  if (!Hessian)
    return(grad)
  
  w = rowSums(W)
  X_jj1 = t(X)%*%(w*X)
  X_jj2 = t(X)%*%(W%*%t(W))%*%(X)
  Hessian = X_jj1 - X_jj2
  
  # for (i in 1:length(indY1))
  # {
  #   WiX = W[,i]*X
  #   X_jj1 = t(X)%*%(WiX)
  #   X_jj2 = t(WiX)%*%WiX
  #   Hessian = Hessian + (X_jj1-X_jj2)
  # }
  
  return(list(grad = grad, hess = Hessian))
}

# calculate the gradient and hessian matrix for Poisson regression given theta
# reference: https://www.stt.msu.edu/users/pszhong/Lecture_10_Spring_2017.pdf
poisson.grad.hess<-function(X, Y, beta, Hessian = F)
{
  p = ncol(X)
  n = nrow(X)
  exp_Xbeta = as.vector(exp(X%*%beta))
  grad = colSums((Y - exp_Xbeta)*X)
  
  if (!Hessian)
    return(grad)
  
  hess = t(X)%*%(exp_Xbeta*X)
  return(list(grad = grad, hess = hess))
}

linear.grad.hess<-function(X, Y, beta, Hessian = F)
{
  p = ncol(X)
  n = nrow(X)
  Xbeta = as.vector(X%*%beta)
  sig2 = mean((Y - Xbeta)^2)
  grad = colSums((Y - Xbeta)*X)/sig2
  if (!Hessian)
    return(grad)
  
  hess = t(X)%*%(X)/sig2
  return(list(grad = grad, hess = hess))
}

get.order.probit.dev<-function(Xbeta, zeta, L, f = dnorm)
{
  #f = dnorm
  f_list = list() # this is partial p_l / partial c
  N = length(Xbeta)
  mat0 = matrix(0, nrow = N, ncol = L-1)
  for (l in 1:L)
  {
    if (l == 1)
    {
      f_list[[1]] = mat0
      f_list[[1]][,1] = f(zeta[1] - Xbeta)
    }
    if (l == L)
    {
      f_list[[L]] = mat0
      f_list[[L]][, L-1] = -f(zeta[L-1] - Xbeta)
    }
    if (l>1&l<L)
    {
      f_list[[l]] = mat0
      f_list[[l]][,l-1] = -f(zeta[l-1] - Xbeta)
      f_list[[l]][,l] = f(zeta[l] - Xbeta)
    }
  }
  return(f_list)
}

# see the file derivatives
order.probit.grad.hess<-function(X, Y, beta, zeta, Hessian = F)
{
  f1 = function(x)
  {
    (2*pi)^{-1/2}*(- x*exp(-x^2/2))
  }
  p = ncol(X)
  n = nrow(X)
  L = ncol(Y)
  xbeta = as.vector(X%*%beta)
  cxbeta = t(outer(zeta, xbeta, "-"))
  
  # prob matrix
  cumprob = pnorm(cxbeta)
  cumprob1 = cbind(cumprob,1)
  probs = cumprob1 - cbind(0, cumprob) # this is p_l
  probs[probs==0] = 10^{-6}
  
  # density matrix
  den = dnorm(cxbeta)
  dens = cbind(den,0) - cbind(0, den) # this is f_l
  
  # gradient
  Y_prob = Y/probs
  Y_den_prob = Y_prob*dens
  grad_beta = -colSums(rowSums(Y_den_prob)*X)
  
  f_list = get.order.probit.dev(xbeta, zeta, L = ncol(Y))
  grad_zeta_mat = Reduce("+" ,lapply(1:L, function(l) Y_prob[,l]*f_list[[l]]))
  grad_zeta = colSums(grad_zeta_mat)
  grad = c(grad_beta, grad_zeta)
  
  if (!Hessian)
    return(grad)
  
  # hessian of beta
  den1 = f1(cxbeta)
  dens1 = cbind(den1,0) - cbind(0, den1) # this is dot f_l
  w1 = rowSums((- dens^2/probs^2 + dens1/probs)*Y)
  hess_beta = t(w1*X)%*%X
  
  # hessian of beta zeta
  f_zeta_list = get.order.probit.dev(xbeta, zeta, L, f = f1)
  w_bz1 = dens/probs^2*Y
  w_bz2 = Y/probs
  
  hess_zeta0 = Reduce("+", lapply(1:L, function(l){
    hbz1 = w_bz1[,l]*f_list[[l]]
    hbz2 = w_bz2[,l]*f_zeta_list[[l]]
    hbz1 - hbz2
  }))
  hess_beta_zeta = t(X)%*%hess_zeta0
  
  # hessian of zeta
  w_z1 = -Y/probs^2
  w_z2 = Y/probs
  hess_zetal = lapply(1:L, function(l){
    hz1 = t(f_list[[l]])%*%(w_z1[,l]*f_list[[l]])
    hz2 = diag(colSums(w_z2[,l]*f_zeta_list[[l]]))
    hz1 + hz2
  })
  hess_zeta = Reduce("+", hess_zetal)
  
  hess_neg = matrix(0, nrow = p+L-1, ncol = p+L-1)
  hess_neg[1:p,1:p] = hess_beta
  hess_neg[1:p,(p+1):(p+L-1)] = hess_beta_zeta
  hess_neg[(p+1):(p+L-1),1:p] = t(hess_beta_zeta)
  hess_neg[(p+1):(p+L-1),(p+1):(p+L-1)] = hess_zeta
  
  hess = -hess_neg
  
  return(list(grad = grad, hess = hess))
}

grad.hess <- function(X, Y, beta, zeta, Hessian = F, reg_type = "logistic")
{
  if (reg_type == 'logistic')
  {
    grad_hess = logistic.grad.hess(X, Y, beta, Hessian = Hessian)
  }
  if (reg_type == 'cox')
  {
    grad_hess = cox.grad.hess(X, Y, beta, Hessian = Hessian)
  }
  if (reg_type == 'poisson')
  {
    grad_hess = poisson.grad.hess(X, Y, beta, Hessian = Hessian)
  }
  if (reg_type == 'linear')
  {
    grad_hess = linear.grad.hess(X, Y, beta, Hessian = Hessian)
  }
  if (reg_type == 'ordered_probit')
  {
    grad_hess = order.probit.grad.hess(X, Y, beta, zeta, Hessian = Hessian)
  }
  return(grad_hess)
}


# CSL method propose by Jordan
# iter.max = 1: one step iteration
# init_oneshot: if True, then use the oneshot estimator as initial
Jordan.distribute<-function(X, Y, K, reg_type = "logistic", 
                            iter.max = 1, init_oneshot = F, ind = NULL)
{
  # data dimension
  p = ncol(X)
  N = nrow(X)
  if (is.vector(Y))
    Y = as.matrix(Y, ncol = 1)
  
  # split the data
  if (is.null(ind))
  {
    ind = rep(1:K, each = floor(N/K)) 
  }
  ns = table(ind) # the sample size of each worker
  
  # set the initial estimator
  if (init_oneshot){
    # use oneshot as initial estimator
    beta_init = dlsa(X, Y, K = K, reg_type)
    theta = beta_init$theta_onehot
    zeta = NULL
  }else{
    # use local estimator on the first worker as the initial estimator
    lr_1 = reg(X = X[ind==1,], Y = Y[ind==1,], reg_type)
    theta = lr_1$theta
    zeta = lr_1$zeta
    theta = c(theta, zeta)
    #hess1 = solve(vcov(lr_1))/ns[1]
  }
  
  del = 1
  iter = 1
  
  theta = tryCatch({
    while (max(abs(del)) > 10^{-4} & iter <= iter.max)
    {
      # cat(iter, del[1:3], "\n")
      # get gradient from local workers
      grad_mat = sapply(1:K, function(k){
        grad.hess(X[ind==k,], Y[ind==k,], beta = theta[1:p], zeta = zeta,
                  Hessian = F, reg_type = reg_type)
      })
      grad = apply(grad_mat, 1, sum, na.rm = T)/sum(ns)
      
      # calculate local hessian matrix
      hess = grad.hess(X = X[ind==1,], Y = Y[ind==1,], 
                       beta = theta[1:p], zeta = zeta, Hessian = T, reg_type)$hess
      hess = hess/ns[1]
      
      del = solve(hess)%*%grad
      if (iter >1 & max(abs(del)) > 5){
        del = del * 0.01
      }
      theta = theta + del
      if (reg_type=="ordered_probit")
      {
        zeta = theta[-(1:p)]
      }
      if (iter == 1)
      {
        theta_onestep = theta
      }
      iter = iter + 1
    }
    return(list(theta = theta[1:p], zeta = zeta, 
                hess = hess, #hess1 = hess1, 
                iter = iter))
  }, error = function(err)
  {
    print(paste("ERROR:  ",err))
    #theta_r = rep(0, p)
    return(list(theta = theta_onestep[1:p], zeta = zeta, 
                hess = hess, #hess1 = hess1, 
                iter = iter))
  })
  
}


# the global onestep estimator, theta_onestep = theta_initial + solve(global_hessian)%*%global_gradient
# this is not distributed estimation
logistic.global.onestep<-function(X, Y, K, init_est = "local", local.hessian = F)
{
  # data dimension
  p = ncol(X)
  N = nrow(X)
  
  # split the data
  ind = rep(1:K, each = floor(N/K)) #sample(1:R, N, replace = T)
  ns = table(ind)
  
  # conduct logistic regression in parallel
  Sig_inv_list = list()
  Sig_inv_theta_list = list()
  
  if (init_est == "oneshot"){
    # use oneshot as initial estimator
    beta_init = logistic.distribute(X, Y, K = K)
    theta = beta_init$theta_onehot
  }else{
    # use local estimator on the first worker as the initial estimator
    lr_1 = logistic(X = X[ind==1,], Y = Y[ind==1], init.beta = NULL, iter.max = NULL)
    theta = lr_1$theta
  }
  
  
  # one-step iteration
  grad_hessian = logistic.grad.hess(X, Y, theta, Hessian = T)
  hess = grad_hessian$hess/N
  grad = grad_hessian$grad/N
  
  
  if (local.hessian)
  {
    # if use local hessian, then replace the hessian with the local estimator
    hess = logistic.grad.hess(X = X[ind==1,], Y = Y[ind==1], theta, Hessian = T)$hess/ns[1]
  }
  
  del = solve(hess)%*%grad
  theta_onestep = theta + del
  return(theta_onestep)
}



