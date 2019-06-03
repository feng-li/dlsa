# Python does not have a good lars package. At the moment we implement this via calling R
# code directly, provided that R package `lars` and python package `rpy2` are both
# installed. FIXME: write a native `lars_las()` function.
import os
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
robjects.r.source(os.path.dirname(os.path.abspath(__file__)) + "/../R/dlsa_alasso_func.R", verbose=False)
lars_lsa=robjects.r['lars.lsa']

# R version
dlsa_r=robjects.r['dlsa']

# Python version
def dlsa(Sig_inv_, beta_, sample_size, intercept=False):
    '''Distributed Least Squares Approximation


    '''

    numpy2ri.activate()
    dfitted = lars_lsa(np.asarray(Sig_inv_), np.asarray(beta_),
                       intercept=intercept, n=sample_size)
    numpy2ri.deactivate()

    AIC = robjects.FloatVector(dfitted.rx2("AIC"))
    AIC_minIdx = np.argmin(AIC)
    BIC = robjects.FloatVector(dfitted.rx2("BIC"))
    BIC_minIdx = np.argmin(BIC)
    beta = np.array(robjects.FloatVector(dfitted.rx2("beta")))


    if intercept:
        beta0 = np.array(robjects.FloatVector(dfitted.rx2("beta0")) + beta[0])
        beta_byAIC = np.concatenate(beta0[AIC_minIdx], beta[AIC_minIdx, :])
        beta_byBIC = np.concatenate(beta0[BIC_minIdx], beta[BIC_minIdx, :])
    else:
        beta_byAIC = beta[AIC_minIdx, :]
        beta_byBIC = beta[BIC_minIdx, :]

    return  pd.DataFrame({"beta_byAIC":beta_byAIC, "beta_byBIC": beta_byBIC})
