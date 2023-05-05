# Python does not have a good lars package. At the moment we implement this via calling R
# code directly, provided that R package `lars` and python package `rpy2` are both
# installed. FIXME: write a native `lars_las()` function.
import os
import zipfile, pathlib

import numpy as np
import pandas as pd

# import rpy2.robjects as robjects
# from rpy2.robjects import numpy2ri
# import rpy2

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

# import pdb
from dlsa.lsa import lars_lsa


def dlsa_mapred(model_mapped_sdf):
    '''MapReduce for partitioned data with given model


    '''
    # mapped_pdf = model_mapped_sdf.toPandas()
    #----------------------------------------------------------------------------------------
    # MERGE
    #----------------------------------------------------------------------------------------
    groupped_sdf = model_mapped_sdf.groupby('par_id')
    groupped_sdf_sum = groupped_sdf.sum(
        *model_mapped_sdf.columns[1:]
    )  # TODO: Error with Python < 3.7 for > 255 arguments. Location 0 is 'par_id'
    groupped_pdf_sum = groupped_sdf_sum.toPandas().sort_values("par_id")

    if groupped_pdf_sum.shape[0] == 0:  # bad chunked models

        raise Exception(
            "Zero-length grouped pandas DataFrame obtained, check the input.")
        # out = pd.DataFrame(columns= ["beta_byOLS", "beta_byONESHOT"] + model_mapped_sdf.columns[3:])

    else:

        Sig_invMcoef_sum = groupped_pdf_sum.iloc[:, 2]
        Sig_inv_sum = groupped_pdf_sum.iloc[:, 3:]

        # beta_byOLS = np.linalg.solve(Sig_inv_sum, Sig_invMcoef_sum)
        beta_byOLS = np.linalg.lstsq(Sig_inv_sum, Sig_invMcoef_sum,
                                     rcond=None)[0]  # least-squares solution

        beta_byONESHOT = groupped_pdf_sum[
            'sum(coef)'] / model_mapped_sdf.rdd.getNumPartitions()
        p = len(Sig_invMcoef_sum)

        out = pd.DataFrame(np.concatenate(
            (beta_byOLS.reshape(p, 1), np.asarray(beta_byONESHOT).reshape(
                p, 1), Sig_inv_sum), 1),
                           columns=["beta_byOLS", "beta_byONESHOT"] +
                           model_mapped_sdf.columns[3:])

    return out


# dlsa_rcode = zipfile.ZipFile(pathlib.Path(__file__).parents[1]).open("dlsa/R/dlsa_alasso_func.R").read().decode("utf-8")
# robjects.r.source(exprs=rpy2.rinterface.parse(dlsa_rcode), verbose=False)
# lars_lsa = robjects.r['lars.lsa']
# dlsa_r = robjects.r['dlsa']

# Python version
def dlsa(Sig_inv_, beta_, sample_size, fit_intercept=False):
    '''Distributed Least Squares Approximation


    '''

    # numpy2ri.activate()
    # dfitted = lars_lsa(np.asarray(Sig_inv_),
    #                    np.asarray(beta_),
    #                    intercept=fit_intercept,
    #                    n=sample_size)
    # numpy2ri.deactivate()

    dfitted = lars_lsa(np.asarray(Sig_inv_),
                       np.asarray(beta_),
                       intercept=fit_intercept,
                       n=sample_size)
    AIC = dfitted["AIC"]
    BIC = dfitted["BIC"]
    beta = dfitted["beta"]
    AIC_minIdx = np.argmin(AIC)
    BIC_minIdx = np.argmin(BIC)

    print(dfitted)
    if fit_intercept:
        beta_byOLS = beta_.to_numpy()

        # beta0 = np.array(robjects.FloatVector(dfitted.rx2("beta0"))) + beta_byOLS[0]
        beta0 = dfitted["beta0"]


        beta_byAIC = np.hstack([beta0[AIC_minIdx], beta[AIC_minIdx, :]])
        beta_byBIC = np.hstack([beta0[BIC_minIdx], beta[BIC_minIdx, :]])
    else:
        beta_byAIC = beta[AIC_minIdx, :]
        beta_byBIC = beta[BIC_minIdx, :]

    return pd.DataFrame({"beta_byAIC": beta_byAIC, "beta_byBIC": beta_byBIC})
