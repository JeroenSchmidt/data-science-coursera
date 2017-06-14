---
typora-copy-images-to: Images
---

# R Language

[TOC]

## Week 1

### Environment Basics

1. R Language
2. R tools (for windows)
3. R Studio

#### Basic Commands

```R
# Get Working Directory
getwd()

# Changing Working Directory
setwd("<directory address>")

# See contents in working directory
dir()

#Load R file
source("<file name>")

# See loaded functions
ls

# Getting help
?<|Command name|>
# for operatiors
?"<|Command name|>"
```

**Note**: every time you make a change to an R file you need to reload it with `source`

### Draw Backs

- Little 3-D graphics support
- Functionality is based on consumer demand
- Wasn't created with parallel computing in mind
- Objects must be stored in physical memory.
  - Work abounds exist

### R Design

**Two Parts**

1. Base R  - controlled by the *CORE group*
2. Everything Else

*Base R Packages*  

`utils`,`stats`,`datasets`,`graphics`,`grDevices`,`grid`,`methods`,`tools`

,`parallel`,`comiler`,`splines`,`tcltk`,`stats4` 

*Recommended Packages*

`boot`, `class`, `cluster`, `codetools`, `foreign`, `KernSmooth`, `lattice`,`mgcv`,`nime`,`rpart`,`survival`,`MASS`,`spatial`,`nnet`,`Matrix`

***NOTE:***

There are +4000 packages on CRAN. These packages are not controlled by the core group, but CRAN has certain standards that need to be meet for a package to be included on CRAN. $\rightarrow$ quality 

### Resources

*Literature:* [List of books for R](http://www.r-project.org/doc/bib/R-books.html)

*Mailing list:* [r-help@r-project.org](mailto:r-help@r-project.org)

#### Asking questions

- What steps will reproduce the problem?
- What is the expected output?
- What do you see instead?
- What version of the product(R,packages, etc) are you using?
- OS?
- Additional info?

*Question Examples*

![img](.\Images\Capture.png)

*How to:*

- Describe the goal
- Explicit in question
  - min amount of information 


### Data types

#### 5 atomic objects

1. character
2. numeric
3. interger
4. complex
5. boolean / logical

***Numbers***

```R
# Explicit declaration of an Integer
x <- 1L
```

*Infinity (i.e. $\frac{1}{0}$)*

```R
Inf
```

*NaN (i.e. $0.0$)*

```R
NaN
```
#### Object Attributes

All objects contain the following attributes

1. Names , dimnames
2. dimensions
3. class
4. length
5. other meta data

```R
# Attributes can be set/modified via
attributes()
```
#### Vector object

- Can only contain objects of the same class
- `List` can contain objects of different classes

*Creating Vectors*

```R
x <- c(item1,item2, etc)
```

Empty vector

```R
x <- vector(<|"data_type"|>, length = #length)
```


*Mixing Objects:* Coercion occurs s.t. every element in the vector is of the same class.

*Explicit Coercion:* Converts the variable from one data type to another

```R
as.<|data_type|>
```

***NOTE:*** When coercion is done, there are cases where a conversion is successfully executed but not necessarily correct.

```R
> x <- c("a","b","c")
> as.numeric(x)
> [1] NA NA NA
> Warning Message
```

#### Lists

Lists are a type of vector that contains elements of different classes

```R
# Declaring a list
x <- list(object1,object2,object3) 
```

#### Matrix

==NOTE:== Many ways to declare a matrix

```R
# Declare a matrix
m <- matrix(nrow = 2, ncol = 3)
```

```R
# Dimension of matrix
dim(#matrix)
#OR
attributes(<|matrix|>)  
```

**Construction**
Matrices are constructed colum-wise. Start at (1,1) and run down the columns.

```R
> m <- matrix(1:6, nrow=2, ncol=3)
> m
     [,1] [,2] [,3]
[1,]    1    3    5
[2,]    2    4    6
```

**Direct Construction**
Create a vector, then add the dimension attribute which changes it to a matrix. 

```R
m <- 1:10
dim(m) <- c(2,5)
```

**Bind Construction**
1) *Column binding*
Constructs columns by putting vectors next to each other as coloumns

```R
cbind(<|vector1|>,<|vector2|>)
```

*2) Row binding*
Constructs columns by putting vectors ontop of each other as rows

```R
rbind(|<vector1|>,<|vector2|>)
```

#### Factors

Represent categorical data. Can be ordered / unordered.
Think of a factor being an integer vector; each integer assigned a label *(refered to as levels)*

Modeling functions like:

- `lm()` - linear model
- `glm()` - general linear model

```R
#Count label
table(#factor)
```

```R
#Separate the object into its components
unclass()
# Note: In the case of factors; itll show you the underlaying integer vector and the label mapping.
```

```R
#Declaring levels(labels)
x <- factors(c("yes","no","yes"), levels = c("yes","no"))
```

*==NOTE:==* In the background factors automatically assigned the level according to alphabetical order. By declaring the *levels* like this we are telling R to start with "yes".

### Missing Values

```R
# Undefined math operations
NaN
```

```R
#Everything Else
Na
```

#### Logic & Cavets

```R
is.na()
is.nan()
```

==NOTE:== 

- NA also has a class 
  - ie NA could look the same but actually isn't the same type of NA under the hood
  - NA (int) vs NA (String)
- NaN is a NA but NA is not a NaN

### DataFrame

Used to store tabular data

Each col is a different object

```R
# Convert DataFrame to Matrix
data.matrix()
#NOTE: Forces each object to be the same
```

```R
#Declaring a Dataframe
x <- data.frame(col1 = #vector, col2= #vector)
```

```R
# Declaring Dataframe row names
rownames(x) <- <vector with row names>
```

```R
# Otherways to create dataframes
read.table()
# OR
read.csv()
```

#### Reading from DataFrames

*DataFrame coloumns*

```R
x <- data.frame(col1 = <values>, col2 = <values>)

## return a vector of values stored in col1
x$col1
#or
x[["col1"]]

## return dataframe with col values
x["col1"]
```

*DataFrame rows*

```R
x <- data.frame(col1 = <values>, col2 = <values>)

## return 1st row using default row names
x[1,]

## return assigend row names
rownames <- 11:20
x["11",]
```

### Name Attributes

```R
# Naming vector entries
x <- vector()
names(x) <- c("name1","name2",...)
```

```R
# Naming list entries
x <- lsit(a=1,b=2,c=3)
# Item 1 is called a, item 2 is called b, etc
```

**Naming Matrix - dimnames**

```R
m <- matrixdimnames(m) <- list(c("row1","row2"), c("col1","col2"))
```

### Reading Data

```R
# Tabluar data
read.table
```

- skips lines with #
- figure out variable types
- CAN specify arguments

```R
read.csv
# Identical to read.table but default separator is a comma
```

```R
# Reading lines of a text file
readLines
```

```R
# Reading R code files
source
```

```R
# Reading R code files - specifically R objects
dget
```

```R
# Reading saved workspaces
load
```

```R
# Reading sinlge R object in binary form
unserialize
```

### Textual Formats 

```R
# Output Textual format 
dumping
# OR
dputing

# Outputs all data including meta-data
```

NOTE: 

- Textual data works well with version controle - good way of keeping track of meta changes
- Easier to fix corruptions
- Not space efficiency

**dput function**

```R
> dput(#object,file="#file_name.R")
# Can only be used on a single R object
```

```R
# Reading dput
> new.y <- dget("#file_name.R")
```

**dump function**

```R
> dump(c("object1","object2"), file = "file_name.R"
```

```R
# reading dump
> source("file_name.R")
```

### Interfacing with outside world

**Some basic connections**

```R
file()
gzfile()
bzfile()
url()
```

**Connection Interface**

Usually we don't need to directly deal with the connection interface. But when dealing with non-standard inputs, using a connection interface can be useful. 

```R
> con <- url("#url", "r")
> x <- readLines(con)
> head(x)
```

### Subsettings

Return one or more element of the same class

```R
1) x[#item_number]
2) x[#lower:#upper]
```

```R
# Extract single element
x[[]]
```

```R
# Logical subsetting
x[#logic_object]
```

```R
# Logic object
u <- x > 3
```

#### Subsetting Lists

```R
x <- list(foo = 1:4, bar = 0.6)
```

```R
# Returns a list with the values 
x[1]
# same as
x["foo"]
```

**Returns just the values**

```R
x[[1]]
```

```R
x$bar 
# the same
asx[["bar"]]
```

```R
# Returing multiple entires
x(c(#lower,#upper))
```

**`$` vs `[[]]`** 

- `$` works for literal names
- `[[]]` works on computed name
  - ie you can use a variable containing names

**Reading nested Lists**

```R
#Reading nested Lists
> x <- list(a = list(10,12,14), b=c(3.14, 2.81))

x[[c(1,3)]]
> 14

x[1]]
> 14
```

#### Subsetting Matrices

**Single Elements**

```R
# Return element by row/col coordinate
x[#row,#col]
```

```R
# Return element by list element number
x[#element_number]
  
# NOTE: The above example, a vector is retrieved.
```

```R
# Preserve the properties of matrix (ie return a matrix)
x[#row,#col,drop = FALSE]
```

**Multiple Elements**

```R
# Return whole row
x[#row, ]
```

```R
# Return whole column
x[ , #col]
```

*NOTE:* `drop = FALSE` also works for multiple elements

#### Removing Missing Values

```R
bad <- is.na(x)
> x[!bad]
```

 *i.e.* sublist of entries that aren't NA using a logical vector

```R
# Check multiple lists for missing values
> good <- complete.cases(x,y)
> x[good]
> y[good]
```

*good*  is a logical list that consists of the combined NA comparison of the two vectors

### Partial Matching

```R
#return partially matched objects
x$a
x[["a", exact = FALSE]]
```

### Vector Operations

```R
# Element wise multiplication
x * y

# Element wise division
x / y

# True matrix multiplication
x %*% y
```

### Reading Large Datasets

Don't use `read.table` if the dataset will be larger then available ram

```R
# Specify the arguments - ColClasses
colClasses = c("<|datatype1|>","<|datatype2|>",etc)
# Determine classes from sample size
inital <- read.table("data.txt", nrows=100)
classes <- sapply(initial,class)
AllData <- read.table("data.txt", colClasses = classes)

# Here we read the first 100 rows of the text file. Determine the classes then declare the classes when we import all the rows in the 3rd line.
```

```R
# Determine number of lines
system(wc <|file-path|>)
       
# NOTE: using unix command wc       
```

#### Determining Memory Requirements

1,500,000 rows
120 columns
All entries are numeric (8 bytes)
$$
\text{The above information produces a rough guess of:}
\\
1,500,000 * 120 * 8 \text{ bytes} 
\\
= 1,440,000,000,000 \text{ bytes} 
\\
\text{ STEP: divide by } 2^{20} \text{for MB}
\\
= 1,373.29 MB
\\
\text{Step: divide by  } 2^{10} \text{for GB}
\\
= 1.34 \text{GB}
$$
==Rule of thumb==: double the estimate, to consider overhead



