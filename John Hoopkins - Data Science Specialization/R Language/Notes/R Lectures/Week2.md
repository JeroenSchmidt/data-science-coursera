---
typora-copy-images-to: Images
---

# Week 2

[TOC]

## Control Structures 

```R
# If 
if (<condition>){
  
}else if(<condition>){
  
}else{
  
}
```

*Assign control result to var*

```R
# Can also assign it to a variable
y <- if(){
  <return value>
}  

# alternativly
if(){
  y <- <value>
}else{
  y <- <value>
}
```

### Loop Structures

#### For Loop

```R
for (i in <vector>){
}

#Example
for (i in 1:3){
  
}
```

*Nested Loops*

```R
# Example: Going through a matrix
x <- matrix(1:6,2,3)

for(i in seq_len(nrow(x))){
  for(j in seq_len(ncol(x))){
    println(x[i,j])
  }
}
```

#### While Loop

```R
while(<condition>){
  
}
```

#### Repeat

```R
# Repeates untill break
repeate{
  if (<condition>){
    break
  }
}
```

#### Skipping iteration

```R
for (i in <vector>){
  if(<condition>){
    next
  }
}
```

## Functions

```R
# Example: Defining a function
# Return vector with values above n
# Default value of n=10
above <- function(x,n = 10){
  use <- x > n
  x[use]
}
```

### Argument Matching

Order of argument matching

1. Exact match for named argument
2. Partial Match
3. Position Match

```R
# Example
args(lm)
function (formula, data, subset, weights, na.action, method = "qr", 
    model = TRUE, x = FALSE, y = FALSE, qr = TRUE, singular.ok = TRUE, 
    contrasts = NULL, offset, ...) 
```

***Note:*** The above function requires the user to input the first 5 arguments. The order of the arguments inputted does not matter as long as they are named. e.g. 

```R
lm(data=<somedata>,formula=x*y,na.action=false,weight=2,subset=<somesubset>)
```

 **Lazy Loading** Arguments are lazy loaded. They are evaluated only as needed.

```R
# No Error will occure
f <- function(a,b){
	a^2
}
> f(2)
> 4
```

```R
# Error will only occure once line 4 is hit
f <- function(a,b){
	a^2
	b+2
}
> f(2)
> 4
## Error: argument "b" is missing
```

**`...`Arguments** 

Indicate a variable number of arguments that are passed onto other functions.

- Often used when extending another function - avoid copying entire list of original arguments. 
- More then one argument that we don't know a user will input. Like `paste`

```R
# Example
myplot <- function(x,y,type="l",...){}
  plot(x,y,type=type,...)
}
```

*Note:* 

1) Arguments after `...` must be explicitly matched. If this is not done, R will assume the value is part of the `...` arguments.  

2) `...` can be left empty when calling a function

## Lexical Scoping 

==Note== R uses *Lexical Scoping* - the values of free variables are searched for in the environment in which the function was defined. 

If R encounters a free variable, it will then travel up the environment tree looking for the variable. 
eg. $\text{Function Env defined in} \rightarrow \text{Parent Env} \rightarrow \text{Top-Level Env : usually global / workspace Env}\rightarrow \\ \text{Further down the environment} \rightarrow \text{Empty Env}$ 

$\text{Parent Env}$: Environment function was called in  

Once the $\text{Empty Env}$ has been reached then an error is thrown.

```R
# Ordered search List for Symbols R looks through
Search()
```

==Note== User can define what order packages get loaded. New package loaded will be loaded one spot bellow the *work environment*. Everything else is moved one spot down. 
==Consequence:== When writing code, we can not assume how scoping will behave.



