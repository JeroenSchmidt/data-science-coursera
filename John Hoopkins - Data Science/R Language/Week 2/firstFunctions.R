add2 <- function(x,y){
  x + y
}

#Return vector with values above 10
above10 <- function(x){
  use <- x > 10
  x[use]
}

# Return vector with values above n
# Default value of n=10
above <- function(x,n = 10){
  use <- x > n
  x[use]
}

# Find Means from cols
columnMean <- function(y, removeNA = TRUE){
  nc <- ncol(y)
  
  #Numeric Vector with 0s
  means <- numeric(nc)
  
  for (i in 1:nc){
    means[i] <- mean(y[,i], na.rm = removeNA)
  }
  
  means
}
