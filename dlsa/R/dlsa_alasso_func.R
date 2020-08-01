suppressPackageStartupMessages(require("lars"))
## Least square approximation. This version May, 2019
## Reference Wang, H. and Leng, C. (2006)
## Rewritten by Xuening Zhu
## Comments and suggestions are welcome
##
## Input
## distributed Sigma Inverse and estiamtors
##
## Output
## beta.ols: the MLE estimate
## beta.bic: the LSA-BIC estimate
## beta.aic: the LSA-AIC estimate

lsa.distribute<-function(theta_mat, Sig_inv_list, intercept = 1, n)
{
  K = ncol(theta_mat)
  Sig_inv_theta_list = list()
  for (k in 1:K)
  {
    Sig_inv_theta_list[[k]] = Sig_inv_list[[k]]%*%theta_mat[,k]
  }
  Sig_inv = Reduce("+", Sig_inv_list)
  beta.ols = solve(Sig_inv)%*%Reduce("+", Sig_inv_theta_list)

  l.fit = lars.lsa(Sig_inv, beta.ols, intercept = intercept, n)
  t1 <- sort(l.fit$BIC, ind=T)
  t2 <- sort(l.fit$AIC, ind=T)
  beta <- l.fit$beta
  if(intercept) {
    beta0 <- l.fit$beta0+beta.ols[1]
    beta.bic <- c(beta0[t1$ix[1]],beta[t1$ix[1],])
    beta.aic <- c(beta0[t2$ix[1]],beta[t2$ix[1],])
  }
  else {
    beta0 <- l.fit$beta0
    beta.bic <- beta[t1$ix[1],]
    beta.aic <- beta[t2$ix[1],]
  }

  obj <- list(beta.ols=beta.ols, beta.bic=beta.bic,
              beta.aic = beta.aic)
  obj

}

dlsa.simplify<-function(beta_, Sig_inv_, intercept = 1, sample_size)
{
    ## Simplified version where the inputs are aggregated beta and Sigma inverse.
    l.fit = lars.lsa(Sig_inv_, beta_, intercept = intercept, n = sample_size)
    t1 <- sort(l.fit$BIC, ind=T)
    t2 <- sort(l.fit$AIC, ind=T)
    beta <- l.fit$beta
    if(intercept) {
        beta_byOLS = beta_ # the input beta_ is the WLSE.
        beta0 <- l.fit$beta0+beta_byOLS[1]
        beta_byBIC <- c(beta0[t1$ix[1]],beta[t1$ix[1],])
        beta_byAIC <- c(beta0[t2$ix[1]],beta[t2$ix[1],])
    }
    else {
        beta0 <- l.fit$beta0
        beta_byBIC <- beta[t1$ix[1],]
        beta_byAIC <- beta[t2$ix[1],]
    }

    obj <- list(beta_byBIC=beta_byBIC,
                beta_byAIC = beta_byAIC)
    obj

}


lsa <- function(obj)
{
  intercept <- attr(obj$terms,'intercept')
  if(class(obj)[1]=='coxph') intercept <- 0

  n <- length(obj$residuals)

  Sigma <- vcov(obj)
  SI <- solve(Sigma)
  beta.ols <- coef(obj)
  l.fit <- lars.lsa(SI, beta.ols, intercept, n)

  t1 <- sort(l.fit$BIC, ind=T)
  t2 <- sort(l.fit$AIC, ind=T)
  beta <- l.fit$beta
  if(intercept) {
    beta0 <- l.fit$beta0+beta.ols[1]
    beta.bic <- c(beta0[t1$ix[1]],beta[t1$ix[1],])
    beta.aic <- c(beta0[t2$ix[1]],beta[t2$ix[1],])
  }
  else {
    beta0 <- l.fit$beta0
    beta.bic <- beta[t1$ix[1],]
    beta.aic <- beta[t2$ix[1],]
  }

  obj <- list(beta.ols=beta.ols, beta.bic=beta.bic,
              beta.aic = beta.aic)
  obj
}


###################################
## lars variant for LSA
lars.lsa <- function (Sigma0, b0, intercept,  n,
                      type = c("lasso", "lar"),
                      eps = .Machine$double.eps,max.steps)
{
  type <- match.arg(type)
  b0 = as.vector(b0)
  TYPE <- switch(type, lasso = "LASSO", lar = "LAR")

  n1 <- dim(Sigma0)[1]

  ## handle intercept
  if (intercept) {
    a11 <- Sigma0[1,1]
    a12 <- Sigma0[2:n1,1]
    a22 <- Sigma0[2:n1,2:n1]
    Sigma <- a22-outer(a12,a12)/a11
    b <- b0[2:n1]
    beta0 <- crossprod(a12,b)/a11
  }
  else {
    Sigma <- Sigma0
    b <- as.vector(b0)
  }

  Sigma <- diag(abs(b))%*%Sigma%*%diag(abs(b))
  b <- sign(b)

  nm <- dim(Sigma)
  m <- nm[2]
  im <- inactive <- seq(m)

  Cvec <- drop(t(b)%*%Sigma)
  ssy <- sum(Cvec*b)
  if (missing(max.steps))
    max.steps <- 8 * m
  beta <- matrix(0, max.steps + 1, m)
  Gamrat <- NULL
  arc.length <- NULL
  R2 <- 1
  RSS <- ssy
  first.in <- integer(m)
  active <- NULL
  actions <- as.list(seq(max.steps))
  drops <- FALSE
  Sign <- NULL
  R <- NULL
  k <- 0
  ignores <- NULL

  while ((k < max.steps) & (length(active) < m)) {
    action <- NULL
    k <- k + 1
    C <- Cvec[inactive]
    Cmax <- max(abs(C))
    if (!any(drops)) {
      new <- abs(C) >= Cmax - eps
      C <- C[!new]
      new <- inactive[new]
      for (inew in new) {
        R <- updateR(Sigma[inew, inew], R, drop(Sigma[inew, active]),
                     Gram = TRUE,eps=eps)
        if(attr(R, "rank") == length(active)) {
          ##singularity; back out
          nR <- seq(length(active))
          R <- R[nR, nR, drop = FALSE]
          attr(R, "rank") <- length(active)
          ignores <- c(ignores, inew)
          action <- c(action,  - inew)
        }
        else {
          if(first.in[inew] == 0)
            first.in[inew] <- k
          active <- c(active, inew)
          Sign <- c(Sign, sign(Cvec[inew]))
          action <- c(action, inew)
        }
      }
    }
    else action <- -dropid
    Gi1 <- backsolve(R, backsolvet(R, Sign))
    dropouts <- NULL
    A <- 1/sqrt(sum(Gi1 * Sign))
    w <- A * Gi1

    if (length(active) >= m) {
      gamhat <- Cmax/A
    }
    else {
      a <- drop(w %*% Sigma[active, -c(active,ignores), drop = FALSE])
      gam <- c((Cmax - C)/(A - a), (Cmax + C)/(A + a))
      gamhat <- min(gam[gam > eps], Cmax/A)
    }
    if (type == "lasso") {
      dropid <- NULL
      b1 <- beta[k, active]
      z1 <- -b1/w
      zmin <- min(z1[z1 > eps], gamhat)
      # cat('zmin ',zmin, ' gamhat ',gamhat,'\n')
      if (zmin < gamhat) {
        gamhat <- zmin
        drops <- z1 == zmin
      }
      else drops <- FALSE
    }
    beta[k + 1, ] <- beta[k, ]
    beta[k + 1, active] <- beta[k + 1, active] + gamhat * w

    Cvec <- Cvec - gamhat * Sigma[, active, drop = FALSE] %*% w
    Gamrat <- c(Gamrat, gamhat/(Cmax/A))

    arc.length <- c(arc.length, gamhat)
    if (type == "lasso" && any(drops)) {
      dropid <- seq(drops)[drops]
      for (id in rev(dropid)) {
        R <- downdateR(R,id)
      }
      dropid <- active[drops]
      beta[k + 1, dropid] <- 0
      active <- active[!drops]
      Sign <- Sign[!drops]
    }

    actions[[k]] <- action
    inactive <- im[-c(active)]
  }
  beta <- beta[seq(k + 1), ]

  dff <- b-t(beta)

  RSS <- diag(t(dff)%*%Sigma%*%dff)

  if(intercept)
    beta <- t(abs(b0[2:n1])*t(beta))
  else
    beta <- t(abs(b0)*t(beta))

  if (intercept) {
    beta0 <- beta0-drop(t(a12)%*%t(beta))/a11
  }
  else {
    beta0 <- rep(0,k+1)
  }
  dof <- apply(abs(beta)>eps,1,sum)
  BIC <- RSS+log(n)*dof
  AIC <- RSS+2*dof
  object <- list(AIC = AIC, BIC = BIC,
                 beta = beta, beta0 = beta0)
  object
}

##Note that rq() object implemented the coef()
##but without vcov() implementation. We provide
##a rather simple implementation here.
##This part is written by Hansheng Wang.
vcov.rq <- function(object,...)
{
  q=object$tau
  x=as.matrix(object$x)
  resid=object$residuals
  f0=density(resid,n=1,from=0,to=0)$y
  COV=q*(1-q)*solve(t(x)%*%x)/f0^2
  COV
}
