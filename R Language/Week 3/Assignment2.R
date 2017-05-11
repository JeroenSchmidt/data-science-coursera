## Coursera Intorduction to R - Assignment 2
## The following script contains functions that cache the inverse of a matrix

## Create a cached matrix

makeCacheMatrix <- function(m = matrix()) {
  i <<- NULL
  
  setMatrix <- function(matrix){
    m <<- matrix
    i <<- NULL
  } 
  
  getMatrix <- function() m
  
  setInverse <- function(inverse) i <<- inverse
  getInverse <- function() i
  
  
  list(setMatrix = setMatrix, getMatrix = getMatrix,
       setInverse = setInverse,
       getInverse = getInverse)
}


## Computes the inverse of a cached matrix. If the inverse has already been calculated 
## (and the matrix has not changed), then cacheSolve retrieves the inverse from the cache.

cacheSolve <- function(x, ...) {
  ## Return a matrix that is the inverse of 'x'
  
  i <- x$getInverse()
  if(!is.null(i)) {
    message("getting cached data")
    return(i)
  }
  
  message("computing new inverse")
  data <- x$getMatrix()
  m <- solve(data, ...)
  x$setInverse(m)
  m
}
