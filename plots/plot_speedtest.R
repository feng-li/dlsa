## nworkers = c(1, seq(4, 84, 4))
nworkers = seq(4, 84, 4)

speedarray = array(NA, c(25, 6, length(nworkers)))

irow = 0
for (i in nworkers)
{
    irow = irow + 1
    path = paste("data/speedtest/","speedtest.s", i, ".ec16", sep = "")
    speedarray[,, irow] = as.matrix(read.table(path, header = TRUE, sep = ","))
}


## Data transformation

par(mfrow = c(1, 2), mar = c(5, 5, 1, 1), cex = 0.5)

for (i in 1:dim(speedarray)[1])
{
    if (i == 1){
        plot(nworkers, speedarray[i,3, ],
             ylim = c(min(speedarray[, 3, ]), max(speedarray[, 3, ])),
             xlim = c(1, max(nworkers) + 20),
             type = "l",
             pch = 20,
             col = "blue",
             lwd = 2,
             axes = FALSE,
             xlab = "No. of Workers", ylab = "Data Transforming Cost (sec.)")
    }
    else {
        lines(nworkers, speedarray[i, 3, ],
              type = "l",
              pch = i,
              col = "blue",
              lwd = 2)
    }

    lines(nworkers, speedarray[i,3,],
          type = "p",
          pch = 20,
          col = "blue",
          lwd = 2)

    if(i > 17)
    {
        text(max(nworkers) + 8, mean(speedarray[i,3, ]),
             paste0(speedarray[i,1, 1], " (", utils:::format.object_size(speedarray[i,2, 1], "auto"), ")"), cex = 0.5)
    }
}

## text(max(nworkers) + 8, 0.1, labels = "<100000(< 1 Mb)", cex = 0.5)
## text(max(nworkers) + 8, 40, labels = "obj. length (in-ram size)", cex = 0.8)
axis(1, nworkers, cex.axis = 0.5) # add the axis on bottom
axis(2, seq(min(speedarray[, 3, ]), max(speedarray[, 3, ]), 2), cex.axis = 0.5)



## Communication cost
## par(mar = c(5, 5, 1, 1), cex = 0.5)
whichrow = 0
col = c("black", "blue", "purple", "red")
dataLenUsed = c(3, 15, 24, dim(speedarray)[1])
for (j in dataLenUsed)
{
    whichrow = whichrow + 1
    if (whichrow == 1){
        plot(nworkers , speedarray[j,5,] - speedarray[j,6, ],
             ylim = c(min(speedarray[, 5, ]), max(speedarray[, 5, ])),
             xlim = c(1, max(nworkers) + 20),
             type = "l",
             pch = 20,
             col = col[whichrow],
             lwd = 2,
             axes = FALSE,
             xlab = "No. of Workers", ylab = "Communication Cost (sec.)")
    }
    else {
        lines(nworkers, speedarray[j,5,] - speedarray[j,6, ],
              type = "l",
              col = col[whichrow],
              lwd = 2)
    }

}

## text(max(nworkers) + 8, 0.1, labels = "<100000(< 1 Mb)", cex = 0.5)
## text(max(nworkers) + 8, 40, labels = "obj. length (in-ram size)", cex = 0.8)
axis(1, nworkers, cex.axis = 0.5) # add the axis on bottom
axis(2, seq(min(speedarray[, 5, ]), max(speedarray[, 5, ]), 0.02), cex.axis = 0.5)
legend("left",
       legend = c(paste("Data Size", utils:::format.object_size(speedarray[dataLenUsed[1], 2, 1], "auto")),
                  paste("Data Size", utils:::format.object_size(speedarray[dataLenUsed[2], 2, 1], "auto")),
                  paste("Data Size", utils:::format.object_size(speedarray[dataLenUsed[3], 2, 1], "auto")),
                  paste("Data Size", utils:::format.object_size(speedarray[dataLenUsed[4], 2, 1], "auto"))),
       col = col, lwd = 2, bty = "n", cex = 0.5)


dev.copy2pdf(file = "CommCost.pdf")
