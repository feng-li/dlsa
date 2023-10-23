#! /usr/bin/env python3

import pandas as pd

from sklearn.linear_model import LogisticRegression

# load the CSV as a Spark data frame
pandas_df = pd.read_csv("../data/games-expand.csv")


# run the model on the partitioned data set
x_train = pandas_df.drop(['label'], axis=1)
y_train = pandas_df["label"]
model = LogisticRegression(solver="lbfgs", fit_intercept=False, penalty='none')
model.fit(x_train, y_train)
p = model.coef_.size
par_id = ["p" + str(x) for x in range(p)]

coef = model.coef_

# Extract probability. By default return n-by-kclasses. We only need the first column for
# binary problems.
prob = model.predict_proba(x_train)[:, 0]
Sig_inv = x_train.T.dot(np.multiply((prob*(1-prob))[:,None],x_train)) / prob.size

out_np = np.concatenate((coef.reshape(coef.size,1),Sig_inv),1)

out_pdf = pd.DataFrame(out_np)

out_pdf2 = pd.concat([pd.DataFrame(par_id,columns=["dd"]),out_pdf],1)
## pred = model.predict_proba(x_train)
