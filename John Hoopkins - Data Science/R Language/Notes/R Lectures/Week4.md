---
typora-copy-images-to: Images
---

# Week 4

[TOC]

## `str` function

## Generating Random Numbers

==Important== Setting the random seed number ***ensures reproducibility!***
`set.seed(<number>)` Make sure that functions you run after setting the seed do not change the seed.

For probability functions there are usually 4 functions associated with them:

* **d** - density
* **r** - random number generation
* **p** - cumulative distribution 
* **q** - quantile function 

**Example**

`rnorm` - generate ***random Normal variates*** with given *norm & SD*

`dnorm` - evaluate the ***Normal probability density*** (with a given mean/SD) at a point (or vector of points)

`pnorm` - evaluate the ***cumulative distribution*** function from a *Normal Distribution* 

`rpois` - ***generate random Poisson variants*** with a *given rate*

### Normal Distribution

```R
dnorm(x, mean = 0, sd = 1, log = FALSE)
pnorm(q, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
qnorm(p, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
rnorm(n, mean = 0, sd = 1)
```

**Arguments**

`x, q` - vector of quantiles.
`p` - vector of probabilities.
`n` - number of observations. If length(n) > 1, the length is taken to be the number required.

If arguments aren't specified: `mean=0` & `sd=1`

==Note==
If $\Phi$ is the cumulative distribution function for a standard Normal distribution, then $\text{pnorm(q)} = \Phi(q)$ and $\text{qnorm(p)} = \Phi^-1(p)$

### Poisson Data

```R
# Example
ppois(2,2) ## probability x <= 2 with distribution 2
```

## Simulating a Linear Model

### Example 1

Say we want to simulate from the following linear model:

$y = \beta_0 + \beta_1x+\epsilon $ 
where $\epsilon \sim N(0,2^2)$ noise with standard distribution with standard deviation of 2
Assume $x \sim N(0,1^2)$ standard *normal* distribution
 $\beta_0 = 0.5$ , $\beta_1 =2$

```R
# Example
set.seed(20)
x <- rnorm(100)
e <- rnomr(100,0,2)
y <- 0.5 + 2*x + e
```

**Plot**

![linearModel](C:\Users\jeroen.schmidt\Documents\Notes\Coursera Notes\R Language\Images/linearModel.PNG)

### Example 2

We can generate similar data for **binary data** by using the binomial function `rbinom`?

```R
e <- rnorm(100,0,2)
y <- 0.5 + 2*x + e
set.seed(20)
x <- rbinom(100,1,0.5)
e <- rnorm(100,0,2)
y <- 0.5 + 2*x + e
plot(x,y)
```

**Plot**

![binarySimulation](C:\Users\jeroen.schmidt\Documents\Notes\Coursera Notes\R Language\Images/binarySimulation.PNG)

### Example 3 - Generalized Linear Model

Say we want to simulate a linear model with a *poisson* distribution.
i.e. $Y \sim \text{Poisson}(\mu)$
$\log(\mu) = \beta_0 + \beta_1x$ where $\beta_0 = 0.5$ and $\beta_1 =0.3$

```R
set.seed(1)
x <- rnorm(100)
log.mu <- 0.5 + 0.3 * x
y <- rpois(100,exp(log.mu))
```

**Plot**

![PoissonSimulation](C:\Users\jeroen.schmidt\Documents\Notes\Coursera Notes\R Language\Images/PoissonSimulation.PNG)

## Random Sampling

```R
# sample 4 values with out replacment
sample(1:10, 4)

# sample with replacment - allows repeats
sample(1:10, replace = TRUE)
```

==NOTE== `sample` changes the seed every time it runs.  Redeclaring the seed every time you run sample resolves this. `set.seed(42); sample(LETTERS, 5)`

## Profiling R Code

*Profiling* is a systematic way of examining how much time is spend on sections of a program.

> Premature optimization is the root of all evil
>
> -Donal Knuth

> Design first, then optimize

 ### Timing

`system.time()` and `proc_time` are used to time code execution times. 

**Two concepts to know**

`Elapse time` - amount of time from start of execution till results returned

`user time` - amount of time the cpu actively handles the program

*Possible outcomes*

`user time` > `elapsed time` : machine has multi core processors 

`elapsed time` >`user time`  : cpu is being occupied by other processes. 

```R
# Example
system.time(<function>)
  
# Longer expressions -> use anon-functions
system.time({
  <code>
})
```

### The Profiler

What if we don't know where to look for bottle necks? We use the *R Profiler* `Rprof()` 
For summaries use `summaryRprof()`

==NOTE== Do not use `system.time()` with `Rprof()`

The default sampling interval is 0.02 seconds, *if the code runs very quickly the profiler isnt that useful*. Use it with code that takes an order of seconds to run.

#### `summaryRprof()`

**Methods that normalize the data**

* `by.total` - divides time spend in each function by the total run time
* `by.self` - does the same but 1st subtracts out time spent in functions above in the call stack

```R
# example
Rprof("name of profile")
## some code to be profiled
Rprof(NULL) # stops profiler
## some code NOT to be profiled

Rprof("name of profile", append=TRUE)
## some code to be profiled
Rprof(NULL) # stops profiler
 
# summarize the results
summaryRprof("name of profile")
```

==NOTE== `C` and `Fortran` code is not profiled