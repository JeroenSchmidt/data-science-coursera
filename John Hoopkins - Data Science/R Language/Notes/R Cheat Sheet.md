[TOC]

# Levels

## Removing Levels not being used

```R
> droplevels(x)
# OR
> subdf$letters
[1] a b c
Levels: a b c d e
> subdf$letters <- factor(subdf$letters)
# OR 
factor(ff) 
```
# Factors

## Converting: Factor -> Numeric

[See link](https://stackoverflow.com/questions/3418128/how-to-convert-a-factor-to-an-integer-numeric-without-a-loss-of-information)

Using `as.Numeric` will convert the Factor keys. *Solution:*

```R
as.numeric(levels(f))[f]
```

OR

```
as.numeric(as.character(f))
```

where `f` is your dataset/dataframe$col

## Concatenating

[See Link](https://stackoverflow.com/questions/3443576/how-to-concatenate-factors-without-them-being-converted-to-integer-level)

Say we concatenate two factor `f1,f2`variables with: `c(f1,f2)` this will coerce the output into a list of integers

eg

```R
> facs <- as.factor(c("i", "want", "to", "be", "a", "factor", "not", "an", "integer"))
> facs
[1] i       want    to      be      a       factor  not     an      integer
Levels: a an be factor i integer not to want
> c(facs[1 : 3], facs[4 : 5])
[1] 5 9 8 3 1
```

***Solution***

```R
unlist(list(f1,f2))
```

## Appending 

[See Link](https://stat.ethz.ch/pipermail/r-help/2008-March/157244.html)

If we try append some value to a factor, the result will be coerced to numeric. 

eg: 

```R
> (a <- factor(c("I","C","I","C","F","I")))
[1] I C I C F I
Levels: C F I
> append(a,"P")
[1] "3" "1" "3" "1" "2" "3" "P"
```

***Solution***

```R
f1 <- factor(c("I","C","I","C","F","I"))
factor(append(as.character(f1),"P"))
```