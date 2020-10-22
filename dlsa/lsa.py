#! /usr/bin/env python3

import numpy as np
from math import log
from math import sqrt


def backsolvet(r, x):
    return (np.linalg.solve(np.triu(r, 0).T, x))


def updateR(xnew, xold, R=None, eps=np.finfo(np.float).eps):
    norm_xnew = np.sqrt(xnew)
    if R is None:
        R = np.matrix(norm_xnew)
        setattr(R, "rank", 1)
        return (R)
    r = backsolvet(R, xold)
    rpp = norm_xnew**2 - sum(r**2)
    if (hasattr(R, "rank")):  ### check if R is machine singular
        rank = getattr(R, "rank")
    else:
        rank = np.NAN
    if (rpp <= eps):
        rpp = eps
    else:
        rpp = np.sqrt(rpp)
        rank = rank + 1
    R = np.column_stack((np.row_stack(
        (R, np.zeros(np.shape(R)[1]))), np.append(r, rpp)))
    setattr(R, "rank", rank)
    return (R)


def delcol(r, z, k):
    p = np.shape(r)[0]
    r = np.delete(r, k, axis=1)
    z = np.matrix(z).T
    nz = np.shape(z)[1]
    p1 = p - 1
    i = k + 1
    while (i < p):
        a = r[i - 1, i - 1]
        b = r[i, i - 1]
        if (not (b == 0)):
            if (not (abs(b) > abs(a))):
                tau = -b / a
                c = 1 / sqrt(1 + tau * tau)
                s = c * tau
            else:
                tau = -a / b
                s = 1 / sqrt(1 + tau * tau)
                c = s * tau
            r[i - 1, i - 1] = c * a - s * b
            r[i, i - 1] = s * a + c * b
            j = i + 1
            while (j <= p1):
                a = r[i - 1, j - 1]
                b = r[i, j - 1]
                r[i - 1, j - 1] = c * a - s * b
                r[i, j - 1] = s * a + c * b
                j = j + 1
            j = 1
            while (j <= nz):
                a = z[i - 1, j - 1]
                b = z[i, j - 1]
                z[i - 1, j - 1] = c * a - s * b
                z[i, j - 1] = s * a + c * b
                j = j + 1
        i = i + 1
    return (r)


def downdateR(R, k):
    p = np.shape(R)[1]
    if (p == 1):
        return (None)
    R = np.delete(delcol(R, np.ones(p), k), p - 1, axis=0)
    setattr(R, "rank", p - 1)
    return (R)  # Built-in Splus utility


#####################################################################
# lars variant for LSA
# Inuput
# Sigma0: Positive definite p-by-p Hessian Matrix
# b0: length p vector
# type:'lar' or 'lasso'
#####################################################################
def lars_lsa(Sigma0,
             b0,
             intercept=True,
             type='lar',
             eps=np.finfo(np.float).eps,
             max_steps=None):
    """
	Compute Least Angle Regression or Lasso path using LARS algorithm [1].
	"""
    n = np.shape(Sigma0)[0]

    #handle intercept
    if (intercept):
        a11 = Sigma0[0, 0]
        a12 = np.array([Sigma0[i, 0] for i in range(1, n)])
        a22 = Sigma0[1:, 1:n]
        Sigma = a22 - np.outer(a12, a12) / a11
        b = b0[1:]
        beta0 = np.dot(a12, b) / a11
    else:
        Sigma = Sigma0
        b = b0
    Sigma = np.diag(np.abs(b)) * Sigma * np.diag(np.abs(b))
    b = np.sign(b)
    nm = np.shape(Sigma)
    m = nm[1]
    im = inactive = np.array(range(1, m + 1))

    Cvec = np.array(b * Sigma)[0]
    RSS = np.sum(Cvec * b)
    if max_steps is None:
        max_steps = 8 * m
    beta = np.matrix(np.zeros((max_steps + 1, m)))
    first = np.zeros(m)
    active = np.array([])
    drops = False
    Sign = []
    R = None
    k = 0
    ignores = []
    while ((k < max_steps) and (len(active) < m)):
        k = k + 1
        C = Cvec[inactive - 1]
        Cmax = max(abs(C))
        if (not np.array(drops).any()):
            new = inactive[abs(C) >= Cmax - eps]
            C = C[abs(C) < Cmax - eps]
            for inew in new:
                R = updateR(Sigma[inew - 1, inew - 1],
                            np.array(Sigma[inew - 1,
                                           (active - 1).astype(int)])[0],
                            R,
                            eps=eps)
                if (getattr(R, "rank") == len(active)):
                    ##singularity; back out
                    nR = np.array(range(len(active)))
                    R = R[nR - 1, nR - 1]
                    setattr(R, "rank", len(active))
                    ignores = np.append(ignores, inew)
                else:
                    if (first[inew - 1] == 0):
                        first[inew - 1] = k
                    active = np.append(active, inew).astype(int)
                    Sign = np.append(Sign, np.sign(Cvec[inew - 1]))

        Gi1 = np.linalg.solve(np.triu(R, 0), backsolvet(R, Sign))
        A = 1 / np.sqrt(sum(Gi1 * Sign))
        w = A * Gi1
        if (len(active) >= m):
            gamhat = Cmax / A
        else:
            a = np.array(
                w * np.delete(Sigma,
                              (np.append(active, ignores) - 1).astype(int),
                              axis=1)[active - 1, ])[0]
            gam = np.append((Cmax - C) / (A - a), (Cmax + C) / (A + a))
            gamhat = min(np.append(gam[gam > eps], Cmax / A))

        if (type == "lasso"):
            dropid = None
            b1 = beta[k - 1, active - 1]
            z1 = -b1 / w
            zmin = np.min(np.append(np.array(z1[z1 > eps])[0], gamhat))
            if (zmin < gamhat):
                gamhat = zmin
                drops = np.array(z1 == zmin)[0]
            else:
                drops = False

        beta[k, ] = beta[k - 1, ]
        beta[k, active - 1] = beta[k, active - 1] + gamhat * w
        Cvec = Cvec - np.array(gamhat * np.dot(Sigma[:, active - 1], w))[0]

        if (type == "lasso" and np.array(drops).any()):
            dropid = np.array(np.where(drops == True))[0]
            for id in dropid[::-1]:
                R = downdateR(R, id)
            dropid = active[drops]
            beta[k, dropid - 1] = 0
            active = active[drops == False]
            Sign = Sign[drops == False]

        inactive = np.delete(im, active - 1)

    beta = beta[np.array(range(k + 1)), ]
    dff = b.reshape(-1, 1) - beta.T
    RSS = np.diag(dff.T * Sigma * dff)

    if (intercept):
        beta = np.multiply(
            np.repeat(np.matrix(abs(b0[1:n])).T, np.shape(beta)[0], axis=1),
            beta.T).T
    else:
        beta = np.multiply(
            np.repeat(np.matrix(abs(b0)).T, np.shape(beta)[0], axis=1),
            beta.T).T

    if (intercept):
        beta0 = beta0 - np.array(a12.T * beta.T)[0] / a11
    else:
        beta0 = np.zeros(k + 1)

    dof = np.array((abs(beta) > eps).sum(1).T)[0]
    BIC = RSS + log(n) * dof
    AIC = RSS + 2 * dof
    object = {'AIC': AIC, 'BIC': BIC, 'beta': beta, 'beta0': beta0}
    return (object)
