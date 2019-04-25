speedarray = array(NA, c(23, 5, 9))

for(i in nworkers)
    {
        path = paste("data/speedtest/","speedtest.s", i, sep = "")
        speedarray[,, i] = as.matrix(read.table(path, header = TRUE, sep = ","))
    }

par(mar = c(5, 5, 1, 1), cex = 0.5)
for(i in 1:23)
{
        if(i == 1){
            plot(1:9, speedarray[i,3, 1:9],
                 ylim = c(min(speedarray[, 3, ]), max(speedarray[, 3, ]) + 1),
                 xlim = c(1, 12),
                 type = "b",
                 pch = 20,
                 col = "blue", lwd = 2,
                 axes = FALSE,
                 xlab = "No. of Workers", ylab = "Communication Cost (sec.)")
        }
        else {
            lines(1:9, speedarray[i,3, 1:9],
                  type = "b",
                  pch = 20,
                  col = "blue", lwd = 2)
        }

        if(i > 17)
        {
            text(10.6, mean(speedarray[i,3, 1:9]),
                 paste0(speedarray[i,1, 1], " (", utils:::format.object_size(speedarray[i,2, 1], "auto"), ")"), cex = 0.5)
        }
}

text(10.7, 0.1, labels = "<100000(< 1 Mb)", cex = 0.5)
text(10.7, 11, labels = "obj. length (in-ram size)", cex = 0.8)
axis(1, 1:9) # add the axis on bottom
axis(2, 0:10)

dev.copy2pdf(file = "CommCost.pdf")
