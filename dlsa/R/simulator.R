library(MASS)
library(survival)

## the simulator for covariates X
simu.X<-function(N, p, K, iid = T)
{
  if (iid)
    X = matrix(rnorm(N*p), ncol = p)
  else{
    nks = rep(floor(N/K), K-1)
    nks[K] = N - sum(nks)
    mus = seq(-1, 1, length.out = K)
    mu_mat = rep(1, p)%*%t(mus)
    sigs = seq(0.3, 0.4, length.out = K)
    X_list = lapply(1:K, function(i){
      XSigma = sigs[i]^abs(outer(1:p,1:p,"-"))
      Xi = mvrnorm(n = nks[i], mu = mu_mat[,i], Sigma = XSigma)
    })
    X = do.call(rbind, X_list)
  }
  return(X)
}

## simulate the Y according to regression model
simu.Y<-function(X, beta, reg_type = "logistic")
{
  if (reg_type == "logistic")
  {
    prob = exp(X%*%beta)/(1+exp(X%*%beta))
    Y = rbinom(N,size = 1, prob)
  }
  if (reg_type == "cox")
  {
    n = nrow(X)
    xt <- X%*%beta
    yt = rexp(n, exp(xt))
    ut <- rexp(n, 1/(runif(n)*2+1)*exp(-xt))
    Y <- Surv(pmin(yt,ut),yt<=ut)
  }
  if (reg_type == "linear")
  {
    n = nrow(X)
    xbeta <- X%*%beta
    Y <- xbeta + rnorm(n)
  }
  if (reg_type == "poisson")
  {
    n = nrow(X)
    xbeta <- X%*%beta
    lamb = exp(xbeta)
    Y = rpois(n, lamb)
  }
  if (reg_type == "ordered_probit")
  {
    n = nrow(X)
    xbeta <- as.vector(X%*%beta)
    cutoff = c(-1, 0, 0.8)
    cxbeta = t(outer(cutoff, xbeta, "-"))
    cumprob = pnorm(cxbeta)
    cumprob1 = cbind(cumprob,1)
    probs = cumprob1 - cbind(0, cumprob)
    
    Y = t(apply(probs, 1, function(x) rmultinom(1, size = 1, prob = x)))
  }
  
  return(Y)
}
