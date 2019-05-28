#! /usr/bin/env python3

import pandas as pd

from sklearn.linear_model import LogisticRegression

# load the CSV as a Spark data frame
pandas_df = pd.read_csv("../data/games-expand.csv")


# run the model on the partitioned data set
x_train = pandas_df.drop(['label'], axis=1)
y_train = pandas_df["label"]
model = LogisticRegression(solver="lbfgs", fit_intercept=False)
model.fit(x_train, y_train)
model.coef_
pred = model.predict_proba(x_train)
