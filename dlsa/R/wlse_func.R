library(survival)
library(MASS)


## Sig_inv of logistic regression
logistic.Sig_inv<-function(X, beta)
{
  N = nrow(X)
  Xbeta = X%*%beta
  prob = as.vector(exp(Xbeta)/(1+exp(Xbeta)))
  Sig_inv = t(X)%*%(prob*(1-prob)*X)/N
  return(Sig_inv)
}

## Sig_inv%*%theta
logistic.Sig_inv_theta<-function(X, beta)
{
  Sig_inv = logistic.Sig_inv(X, beta)
  return(Sig_inv%*%beta)
}


## logistic regression
logistic<-function(X, Y, init.beta = NULL, iter.max = NULL)
{
  N = nrow(X)
  p = ncol(X)
  if (is.null(init.beta))
  {
    beta = rep(0, p)
  }else{
    beta = init.beta
  }
  del = 1
  iter = 0
  if (is.null(iter.max)){
    iter.max = 80
  }
  while(max(abs(del)) > 10^{-4} & iter < iter.max)
  {
    #cat(max(abs(del)), "\n")
    #if (max(abs(del)) > 0.1){
    #  cat("Not Converge! \n")
    #  beta = runif(p, -1, 1)
    #}
    # calculate obj function
    Xbeta = X%*%beta
    prob01 = 1/as.vector(1+exp(Xbeta))*cbind(1, exp(Xbeta))
    obj = sum(Y*log(prob01[,2]))+sum((1-Y)*prob01[,1])
    
    # obtain gradient and Hessian matrix
    prob = as.vector(exp(Xbeta)/(1+exp(Xbeta)))
    grad = colSums((Y - prob)*X)
    hess = t(X)%*%(prob*(1-prob)*X)
    del = solve(hess)%*%grad
    if (max(abs(del)) > 5){
      #cat("Not Converge! \r")
      #cat(obj, "\n")
      del = del * 0.01
    }
    beta = beta + del
    iter = iter + 1
  }
  return(list(theta = beta, Sig_inv = hess, iter = iter))
}


# the global function
# reg_type = 'logistic', 'cox', 'poisson', 'order', 'linear'
reg<-function(X, Y, reg_type = "logistic")
{
  if (reg_type == 'logistic')
  {
    obj = logistic(X, Y)
  }
  if (reg_type == "cox")
  {
    obj = coxph(Y~X-1)
    obj$theta = coef(obj)
  }
  if (reg_type == 'linear')
  {
    obj = lm(Y~X-1)
    obj$theta = coef(obj)
  }
  if (reg_type == "poisson")
  {
    obj = glm(Y~X-1, family = "poisson")
    obj$theta = coef(obj)
  }
  if (reg_type == "ordered_probit")
  {
    if (!is.null(dim(Y)))
    {
      lev = 1:ncol(Y)
      Y_fac = colSums(lev*t(Y))
      #Y_fac = apply(Y, 1, which.max)
    }else{
      if (is.factor(Y))
        Y_fac = Y
      lev = sort(unique(Y_fac))
    }
    Y_fac = factor(Y_fac, levels = lev)
    obj = polr(Y_fac~X, Hess = T, method = "probit")
    obj$theta = coef(obj)
  }
  return(obj)
}


# distributed logistic regression
# K is number of workers
dlsa<-function(X, Y, K, reg_type = "logistic",
               init.beta = NULL, iter.max = NULL, ind = NULL)
{
  # data dimension
  p = ncol(X)
  N = nrow(X)
  
  # split the data
  if (is.null(ind))
  {
    ind = rep(1:K, each = floor(N/K)) #sample(1:R, N, replace = T)
  }
  
  
  # conduct logistic regression in parallel
  Sig_inv_list = list()
  Sig_inv_theta_list = list()
  
  theta_mat = matrix(0, nrow = p, ncol = K)
  for (k in 1:K)
  {
    if (reg_type=="logistic")
    {
      lr_k = logistic(X = X[ind==k,], Y = Y[ind==k], init.beta, iter.max)
      Sig_inv_list[[k]] = lr_k$Sig_inv
      Sig_inv_theta_list[[k]] = lr_k$Sig_inv%*%lr_k$theta
      theta_mat[,k] = lr_k$theta
    }
    if (reg_type=="cox")
    {
      y = Y[ind==k]
      x = X[ind==k,]
      cox_k = coxph(y~x-1)
      
      Sigma <- vcov(cox_k)
      SI <- solve(Sigma)
      beta.ols <- coef(cox_k)
      
      Sig_inv_list[[k]] = SI
      Sig_inv_theta_list[[k]] = SI%*%beta.ols
      theta_mat[,k] = beta.ols
    }
    if (reg_type=="poisson")
    {
      y = Y[ind==k]
      x = X[ind==k,]
      obj = glm(y~x-1, family = "poisson")
      
      Sigma <- vcov(obj)
      SI <- solve(Sigma)
      beta.ols <- coef(obj)
      
      Sig_inv_list[[k]] = SI
      Sig_inv_theta_list[[k]] = SI%*%beta.ols
      theta_mat[,k] = beta.ols
    }
    if (reg_type=="linear")
    {
      y = Y[ind==k]
      x = X[ind==k,]
      obj = lm(y~x-1)
      
      Sigma <- vcov(obj)
      SI <- solve(Sigma)
      beta.ols <- coef(obj)
      
      Sig_inv_list[[k]] = SI
      Sig_inv_theta_list[[k]] = SI%*%beta.ols
      theta_mat[,k] = beta.ols
    }
    if (reg_type=="ordered_probit")
    {
      y = Y[ind==k,]
      x = X[ind==k,]
      obj = reg(x, y, reg_type)
      
      Sigma <- suppressMessages(vcov(obj))
      SI <- solve(Sigma)[1:p,1:p]
      beta.ols <- coef(obj)
      
      Sig_inv_list[[k]] = SI
      Sig_inv_theta_list[[k]] = SI%*%beta.ols
      theta_mat[,k] = beta.ols
    }
  }
  Sig_inv_sum = Reduce("+", Sig_inv_list)
  Sig_inv_theta_sum = Reduce("+", Sig_inv_theta_list)
  theta = solve(Sig_inv_sum)%*%Sig_inv_theta_sum
  return(list(theta = theta, theta_onehot = rowMeans(theta_mat), Sig_inv = Sig_inv_sum,
              theta_mat = theta_mat, Sig_inv_list = Sig_inv_list))
}



## 2 step estimation (using one round communication)
logistic.distribute.2step<-function(X, Y, K)
{
  theta_2step = matrix(0, nrow = p, ncol = 2)
  init.beta = NULL
  for (s in 1:2)
  {
    if (s == 2)
    {
      iter.max = 1
    }else{
      iter.max = 80
    }
    
    theta_est = logistic.distribute(X, Y, K, init.beta, iter.max = iter.max)
    init.beta = theta_est$theta_onehot#theta
    theta_2step[,s] = theta_est$theta
  }
  
  return(list(theta_2step = theta_2step[,2], 
              theta_1step = theta_2step[,1],
              theta_onehot = theta_est$theta_onehot))
}

