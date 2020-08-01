#! /usr/bin/env Rscript

## require("reticulate")
## use_python("/usr/bin/python3")
## source_python("utils_plot.py")

## sgd = "~/running/logistic_sgd_model_2019-07-27-00:08:04.pkl"
## dlsa = "~/running/logistic_dlsa_model_2019-07-26-23:54:15.pkl"


## est_sgd <- read_pickle_file(sgd)
## est_dlsa <- read_pickle_file(dlsa)


coef_matrix = read.table("coef.csv", header = TRUE, sep = ",")

# colnames(coef_matrix) = c("MLE", "DLSA_AIC", "DLSA_BIC", "WLSE", "ONE_HOT")
par(mfrow = c(5, 1), mar = c(2, 5, 2, 0), las = 2)
color = c("blue", "purple", "red", "gray", "cyan")

for (i in 2:6)
{
    # used_data = coef_matrix[-c(which.max(coef_matrix[, i]), which.max(coef_matrix[,
                                        # i])), i]
    used_data = coef_matrix[, i]

    if (i == 4)
    {
        used_data[abs(used_data) < 0.1] = 0
        ## ylim = c(-0.02, 0.02)
    }



    if (i != 2)
    {
        # ylim = c(-0.02, 0.02)
    }
    else
    {
       #  ylim = c(-400, 400)
    }

    ## if (i == 6){
    ## barplot(used_data, ylab = colnames(coef_matrix)[i], ylim = ylim,
    ##         # xlab = coef_matrix[, 1],
    ##         xpd = TRUE,
    ##         las = 2,
    ##         names.arg = 1:182,
    ##         col = "cyan", axes = TRUE)
    ## }
    ## else
    ## {
    barplot(used_data, ylab = colnames(coef_matrix)[i], # ylim = ylim,
            xpd = TRUE,
            las = 2,
            col = "cyan", axes = TRUE)

    ## }
}







 plot(coef_matrix[, 6], coef_matrix[, 5], pch = 20, col = "blue", xlab = "ONE_SHOT", ylab = "WLSE")
