---
typora-copy-images-to: Images
---

# Week 3

***Loop functions*** and debugging

[TOC]

## Looping functions

**Example:** Dealing with function *arguments*
i.e. just add them to the next argument spaces  

```
lapply(x,runif, min = 0, max = 1)
```

### `lapply` - loop over a list and evaluate a function on each element

- always returns a list

### `sapply` - same as `lapply` but try to simplify the result

- if all the elements are of the size 1 -> returns a ***vector***
- if all the elements are of the same size -> returns a ***matrix***
- *else* returns a ***list***

### `apply` - apply a function over a the margins of an array

- apply function to ***rows*** & ***columns***
- Also works on whole arrays
- not faster then using a convention loop

```R
# cal the mean of each col of a matrix
x <- matrix(rnorm(200),20,10) #20 rows, 10 cols dim(20,10)

apply(x,2,mean) #2 -> keep the 2nd dim and colapse the 1st dim into the function -> ie find the mean for each col

#find mean of each row
apply(x,1,mean)
```

```R
# dealing with higher dimensions - 3D cube
a <- arrray(rnorm(2 * 2 * 10), c(2,2,10))

# find the mean along the 3rd dimension, keeping the 1st an 2nd dim
apply(a,c(1,2),mean)

# can also use more optimised function
rowMeans(a, dims =2)
```

#### Optimized apply functions

The functions have been optimized to run ***much*** faster then apply 

*  `rowSums`
*  `rowMeans`
*  `colSums`
*  `colMeans`

### `mapply` - multivariate version of `lapply`

This applied a function parallel over a set of arguments

```R
#Basic example
> mapply(rep,1:4,4:1)
[[1]]
[1] 1 1 1 1

[[2]]
[1] 2 2 2

[[3]]
[1] 3 3

[[4]]
[1] 4
```

***Vectorization of a function***

```R
#Example
noise <- function(n,mean sd){
  rnorm(n,mean,sd)
}

#We can do the following with mapply
list(noise(1,1,2), noise(2,2,2),noise(3,3,2))

# with mapply
mapply(noise,1:3,1:3,2) #changing the n and mean for 3 different results
```

### `tapply`  - apply a function over subsets of a vector

In the background think of the list being split into groups with `split`

```R
# take 3 group normals
x <- c(rnorm(10),runif(10),rnorm(10,1))
f <- gl(3,10) 

tapply(x,f,mean)
         1          2          3 
0.07089416 0.40943126 0.71821110 

# the same as 
lapply(split(x,f),mean)
```

### `split` 

* Divides the data in the vector x into the groups defined by f

==NOTE== has `drop` argument that drops empty levels that are empty

```R
# Spliting to 3 groups
x <- c(rnorm(10),runif(10),rnorm(10,1))
f <- gl(3,10) 

split(x,f)
$`1`
 [1] -1.0510602  0.6390667 -1.1844335  1.4694339 -0.5913725
 [6]  0.4665564 -1.4135527  1.9782572  0.6352339 -0.2391876

$`2`
 [1] 0.250213059 0.468191714 0.555302177 0.495414285 0.162347914
 [6] 0.404373503 0.418992073 0.450538643 0.887675458 0.001263779

$`3`
 [1]  1.6352665  1.0328139  0.5006926 -0.9504895  0.6489016
 [6]  0.6262256  1.9785490  2.2131524 -0.6940379  0.1910369
```

More complicated example

```R
# split the dataframe according to months and find the means for each month
s <- split(airquality, airquality$Month)

sapply(s,function(x) colMeans(x[,c("Ozone","Solar.R","Wind")]), na.rm=TRUE) 
```

#### Splitting on >1 levels

Spit according to the combination of multiple factors 

```R
x <- rnorm(10)
f1 <- gl(2,5) # 2 levels
f2 <- gl(5,2) # 5 levels

interaction(f1,f2) # all combinations of different levels
 [1] 1.1 1.1 1.2 1.2 1.3 2.3 2.4 2.4 2.5 2.5
Levels: 1.1 2.1 1.2 2.2 1.3 2.3 1.4 2.4 1.5 2.5

> split(x,list(f1,f2))
$`1.1`
[1] -0.5988166 -1.4389235
$`2.1`
numeric(0)
$`1.2`
[1] -1.3074090  0.4250326
$`2.2`
numeric(0)
#...etc
```

### Anonymous functions in loop functions

The above listed loop functions make heavy use of *anonymous functions*

```R
# function that gets first col for each matrix
x <- list(1:4,2,2), b = matrix(1:6,3,2)

lapply(x, function(elt) elt[,1])
```

## Debugging

Logging types

* `message`
* `warning`
* `error`
* `condition` - generic concept that programmer can extend to make their own logging type\

### Tools

#### `traceback`  

prints out function call stack

```R
> mean(x)
Error
> traceback() #has to be executed immediately after code is executed 
l: mean(x)
```

#### `debug` 

flags a function for "debug" mode, step through function line at a time> 

```R
> debug(lm)
> lm(y - x)
#debugger executes
# press "n" for next line
```

#### `browser `

Suspends execution of function wherever it is called and puts function into debug mode

#### `trace`

Allows you to insert debugging code into a function at specific places (without editing the function it self)

#### `recover` 

Allows you to modify the error behavior so that you can browse the function call stack

```R
> options(error = recover) #global option
> read.csv("no file")
#debuger exetures but offers options of what to do next
```

## other 

### Creating function where an assignment occurs but nothing is printed 

```R
f1 <- function(x) x
f2 <- function(x) invisible(x)
  
y <- f1(1)
>	1
y <- f2(1) #nothing returned 

y
> 1
```
### formatting outputs of loop functions / formatting lists `unlist`

[Understanding lapply()](https://www.coursera.org/learn/r-programming/discussions/weeks/3/threads/qcv_orZHEeWF2Q53QdZUbw)

[A few pointers for assignment 3](https://www.coursera.org/learn/r-programming/discussions/weeks/4/threads/znVFbLgpEeWlQwoU9G612w)

```R
# say we want somthing like this
                                             hospital state
AK                        FAIRBANKS MEMORIAL HOSPITAL    AK
AL                          SPRINGHILL MEDICAL CENTER    AL
AR          BAPTIST HEALTH MEDICAL CENTER-LITTLE ROCK    AR
AZ                     CARONDELET ST JOSEPHS HOSPITAL    AZ
CA                          HUNTINGTON BEACH HOSPITAL    CA
CO                               MCKEE MEDICAL CENTER    CO
CT                         ROCKVILLE GENERAL HOSPITAL    CT
DC                                               <NA>    DC

# but our lapply produces somthing like this
$AK
[1] "FAIRBANKS MEMORIAL HOSPITAL"

$AL
[1] "SPRINGHILL MEDICAL CENTER"

$AR
[1] "BAPTIST HEALTH MEDICAL CENTER-LITTLE ROCK"

$AZ
[1] "CARONDELET ST JOSEPHS HOSPITAL"

$CA
[1] "HUNTINGTON BEACH HOSPITAL"

$CO
[1] "MCKEE MEDICAL CENTER"

$CT
[1] "ROCKVILLE GENERAL HOSPITAL"

## HOW DO WE FORMAT this list?
```

```R
  ## USE  unlist
  ranks <- lapply(split(orderedDF,orderedDF$State),function(x){
    
    rank <- num  
    if (class(num) == "character"){
      if (num == "best"){
        rank <- 1
      }else if (num == "worst"){
        rank <- dim(x)[1]
      }
    }
    x[rank,"Hospital.Name"]#["Hospital.Name"]
  })
  
  output <- data.frame(hospital=unlist(x,use.names = FALSE),state=names(ranks),row.names=names(ranks))
```