import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings

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

def logistic_model(sample_df, Y_name, fit_intercept=False, convert_dummies=[]):
    '''Run logistic model on the partitioned data set

    '''

    # x_train = sample_df.drop(['label', 'row_id', 'partition_id'], axis=1)
    # sample_df = samle_df.dropna()

    # Special step to create a local dummy matrix
    if len(convert_dummies) > 0:
        X_with_dummies = pd.get_dummies(data=sample_df,
                                        drop_first=fit_intercept,
                                        columns=convert_dummies,
                                        sparse=True)

        x_train = X_with_dummies.drop(['partition_id', Y_name], axis = 1)
        x_train.sort_index(axis=1, inplace=True)

    else:
        x_train = sample_df.drop(['partition_id', Y_name], axis=1)


    y_train = sample_df[Y_name]

    model = LogisticRegression(solver="lbfgs", penalty="none", fit_intercept=fit_intercept, max_iter=500)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_train)[:, 0]
    p = model.coef_.size

    coef = model.coef_.reshape(p, 1) # p-by-1
    Sig_inv = x_train.T.dot(np.multiply((prob*(1-prob))[:,None],x_train)) # p-by-p
    Sig_invMcoef = Sig_inv.dot(coef) # p-by-1

    # grad = np.dot(x_train.T, y_train - prob)

    # Assign par_id
    par_id = pd.DataFrame(np.arange(p).reshape(p, 1), columns=['par_id'])
    # par_id = pd.DataFrame(x_train.columns.to_numpy().reshape(p, 1), columns=["par_id"])

    out_np = np.concatenate((coef, Sig_invMcoef, Sig_inv),1) # p-by-(3+p)
    out_pdf = pd.DataFrame(out_np,
                           columns=pd.Index(["coef", "Sig_invMcoef"] + x_train.columns.tolist()))
    out = pd.concat([par_id, out_pdf],1)

    if out.isna().values.any():
        warnings.warn("NAs appear in the final output")

    return out
    # return pd.DataFrame(Sig_inv)
