specify_decimal = function(x, k) gsub('\\s+','',format(round(x, k), nsmall=k))                                   ### format function for keeping 2 digits after

rowSd<-function(x){
  apply(x, 1, sd)
}

rowMeans.K<-function(x, K = 4){
  specify_decimal(rowMeans(x, na.rm = T), K)
}

rowRMSE.K<-function(x, K = 4){
  specify_decimal(sqrt(rowMeans(x^2, na.rm = T)), K)
}

MSE.ratio.K<-function(x, y, K = 4){
  specify_decimal(sqrt(rowMedian(x^2))/sqrt(rowMedian(y^2)), K)
}

RMSE.ratio.K<-function(x, y, K = 2){
  specify_decimal(sqrt(rowMeans(x^2))/sqrt(rowMeans(y^2)), K)
}

rowMedian<-function(x, na.rm = T){
  apply(x, 1, median, na.rm = na.rm)
}


rowMedian.K<-function(x, K = 4){
  specify_decimal(apply(x, 1, median, na.rm = T), K)
}

rowSd.K<-function(x, K = 4){
  specify_decimal(apply(x, 1, sd, na.rm = T), K)
}
