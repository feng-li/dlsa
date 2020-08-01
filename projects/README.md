# Logistic Regression with DLSA for Airline data


## Dataset

For illustration purposes, we study a large real-world dataset. Specifically, the dataset
considered here is the U.S. Airline Dataset. The dataset is available at
http://stat-computing.org/dataexpo/2009. It contains detailed flight information
about U.S. airlines from 1987 to 2008. 


## Tasks

The task is to predict the delayed status of a flight given all other flight information
with a logistic regression model. Each sample in the data corresponds to one flight
record, which consists of a binary response variable for delayed status `Delayed`, and
departure time, arrival time, distance of the flight, flight date, delay status at
departures, carrier information, origin and destination as regressors.

## Variables

The complete variable information is described in [Zhu, Li and Wang
(2019)](https://arxiv.org/abs/1908.04904).  The data contain six continuous variables and
five categorical variables. The categorical variables are converted to dummies with
appropriate dimensions. We treat the `Year` and `DayofMonth` variables as numerical to
capture the time effects. To capture possible seasonal patterns, we also convert the time
variables `Month` and `DayofWeek` to dummies. Ultimately, a total of 181 variables are
used in the model. The total sample size is 113.9 million observations. This leads to the
raw dataset being 12 GB on a hard drive. After the dummy transformation described in [Zhu,
Li and Wang (2019)](https://arxiv.org/abs/1908.04904), the overall in-memory size is over
52 GB, even if all the dummies are stored in a sparse matrix format. Thus, this dataset
can hardly be handled by a single computer. All the numerical variables are standardized
to have a mean of zero and a variance of one.

## Run with Spark

```sh
sh bash/spark_dlsa_run.sh
```

