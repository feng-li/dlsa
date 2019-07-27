#! /usr/bin/env Rscript

## require("reticulate")
## use_python("/usr/bin/python3")
## source_python("utils_plot.py")

## sgd = "~/running/logistic_sgd_model_2019-07-27-00:08:04.pkl"
## dlsa = "~/running/logistic_dlsa_model_2019-07-26-23:54:15.pkl"


## est_sgd <- read_pickle_file(sgd)
## est_dlsa <- read_pickle_file(dlsa)


coef_matrix = read.table("coef.csv", header = FALSE, sep = ",")

colnames(coef_matrix) = c("MLE", "DLSA_AIC", "DLSA_BIC", "WLSE", "ONE_HOT")
par(mfrow = c(5, 1), mar = c(0, 5, 0, 0))
color = c("blue", "purple", "red", "gray", "cyan")

for (i in c(4, 5, 1, 2, 3))
{
    used_data = coef_matrix[-c(which.max(coef_matrix[, i]), which.max(coef_matrix[, i])), i]
    barplot(used_data, ylab = colnames(coef_matrix)[i], col = "cyan", axes = FALSE)
}
