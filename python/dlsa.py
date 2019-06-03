# Python does not have a good lars package. At the moment we implement this via calling R
# code directly, provided that R package `lars` and python package `rpy2` are both
# installed. FIXME: write a native `lars_las()` function.
import os
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

import pandas as pd
import numpy as np

def dlsa_mapred(model, data_sdf, partition_id):
    '''MapReduce for partitioned data with given model

    '''


    # partition the data and run the UDF
    mapped_sdf = data_sdf.groupby(partition_id).apply(model)

    # mapped_pdf = mapped_sdf.toPandas()
    ##----------------------------------------------------------------------------------------
    ## MERGE
    ##----------------------------------------------------------------------------------------
    groupped_sdf = mapped_sdf.groupby('par_id')
    groupped_sdf_sum = groupped_sdf.sum(*mapped_sdf.columns[1:])
    groupped_pdf_sum = groupped_sdf_sum.toPandas().sort_values("par_id")

    Sig_invMcoef_sum = groupped_pdf_sum.iloc[:,2]
    Sig_inv_sum = groupped_pdf_sum.iloc[:,3:]

    par_byOLS = np.linalg.solve(Sig_inv_sum, Sig_invMcoef_sum)

    par_byONEHOT = groupped_pdf_sum['sum(coef)'] / data_sdf.rdd.getNumPartitions()
    # out_par_onehot = groupped_pdf_sum['sum(coef)'] / partition_num
    p = par_byOLS.size

    return pd.DataFrame(np.concatenate((par_byOLS.reshape(p, 1),
                                        np.asarray(par_byONEHOT).reshape(50, 1),
                                        Sig_inv_sum), 1),
                        columns= ["par_byOLS", "par_byONEHOT"] + ["x" + str(i) for i in range(p)])



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


def simulate_logistic(sample_size, p, partition_method, partition_num):
    '''Simulate data based on logistic model

    '''



    ## Simulate Data
    n = sample_size
    p1 = int(p * 0.4)

    # partition_method = "systematic"
    # partition_num = 200

    ## TRUE beta
    beta = np.zeros(p).reshape(p, 1)
    beta[:p1] = 1

    ## Simulate features
    features = np.random.rand(n, p) - 0.5
    prob = 1 / (1 + np.exp(-features.dot(beta)))

    ## Simulate label
    label = np.zeros(n).reshape(n, 1)
    partition_id = np.zeros(n).reshape(n, 1)
    for i in range(n):
        # TODO: REMOVE loop
        label[i] = np.random.binomial(n=1,p=prob[i], size=1)

        if partition_method == "systematic":
            partition_id[i] = i % partition_num
        else:
            raise Exception("No such partition method implemented!")

        data_np = np.concatenate((partition_id, label, features), 1)
        data_pdf = pd.DataFrame(data_np, columns=["partition_id"] + ["label"] + ["x" + str(x) for x in range(p)])

    return data_pdf
