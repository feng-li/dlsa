import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

def logistic_model(sample_df):
    # run the model on the partitioned data set
    # x_train = sample_df.drop(['label', 'row_id', 'partition_id'], axis=1)
    x_train = sample_df.drop(['partition_id', 'label'], axis=1)
    y_train = sample_df["label"]
    model = LogisticRegression(solver="lbfgs", fit_intercept=False)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_train)[:, 0]
    p = model.coef_.size

    coef = model.coef_.reshape(p, 1) # p-by-1
    Sig_inv = x_train.T.dot(np.multiply((prob*(1-prob))[:,None],x_train)) / prob.size # p-by-p
    Sig_invMcoef = Sig_inv.dot(coef) # p-by-1

    # grad = np.dot(x_train.T, y_train - prob)

    par_id = np.arange(p)

    out_np = np.concatenate((coef, Sig_invMcoef, Sig_inv),1) # p-by-(2+p)
    out_pdf = pd.DataFrame(out_np)
    out = pd.concat([pd.DataFrame(par_id,columns=["par_id"]), out_pdf],1)

    return out
    # return pd.DataFrame(Sig_inv)
