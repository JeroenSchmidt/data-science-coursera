---
typora-copy-images-to: Images
---

# Data Science Tools

[TOC]

------

## Week 1

### Asking the Right Questions

*Google Quarry:*

```
[data type] data analysis
```

|     **Key work to use**     | **Type of data youre looking for** |
| :-------------------------: | :--------------------------------: |
|        Biostatistics        |            Medical data            |
|        Data science         |       Data from web analysis       |
|      Machine learning       |   Data in CS or computer vision    |
| Natural language processing |           Data from text           |
|      Signal processing      |    Data from electrical signals    |
|     Business analytics      |        Data from customers         |
|        Econometrics         |           Economic data            |
| Statistical process control |  Data about industrial processes   |

### Finding help in R

```R
?rnorm #Access help file

help.search("rnorm") #Search help files

args("rnorm") #Get arguments
#returns
-> function(n,mean=0,sd=1) 

#see code
rnorm
#returns the code that corresponds to the function
```

------

#### Where to look for help

*Stackoverflow* - use the tag in search terms

```
"[r]"
```

*Google*

```
[data type] R package
```

## Week 2

### Command Line Interface

Look at "Linux Basics" notes

### Git

#### Create local repository

```shell
mkdir ~/text-repo ##go to directory where repo is to be made
cd ~/test-repo
git init ## make repo
git remote add origin [repo url] ## link to remote repo
git clone [github repo] ## clone a remote repo
```

#### Git commands and structure 

![git](.\Images/git.PNG)

**Adding**

```shell
# Adding new files to local repo
git add . #Add all new files
git add -u #Updates tracking for files that changes names or were deleted
git add -A #Does all of the above
```

**Committing**

```shell
git commit -m "message" #commits all files that have been added
```

**Push**

```shell
# Pushes the commits to the remote repo
git push
```

**Branches**

```shell
git checkout -b [branch name] #create a branch
git brach # see current branch name
git checkout [branch name] # switch to another branch
git branch -a #see all branches 
```

**Pull Request**

Go to github and make a pull request to merge branches together

### Markdown

Text file that R, R studio and github use for documentation 

```markdown
[comment]: <> (Headings) 
## This is a secondary heading
### This is a tertiary heading
```

```markdown
[comment]: <> (Unordered Lists)
* 1st item in list
* 2nd item in list
* 3rd item in list
```

**More material** 

1. www.daringfireball.net/projects/markdown
2. In RStudio click on the MD button for a quick guide 

### RStudio

```R
# See available packages
a <- available.packages()
head(rownames(a),3) ## show the names of the first few packages 
```

List of packages according to [topic]([https://cran.r-project.org/web/views/](https://cran.r-project.org/web/views/))

```R
# Install a package
install.packages(["name"])
install.packages(c("package1","package2",etc))
```

#### Installing R Packages from Bioconductor

```R
# Installs basic set of R packages
source("http://bioconductor.org/bioLite.R") ## Loads the biocLite function
biocLite() ## Loads basic set

#Install specific packages
biocLite(c("GenomicFeatures","AnnotationDbi "))
```

NOTE: Even if packages are installed, they might not be loaded into the environment

#### Loading Packages

```R
# Load package
library([package name]) 
```

NOTE: 

1. Dependent packages will be loaded first before named packages are loaded
2. Don't put package names in quotes 

Libraries that are loaded will show up at the top of the search list

```R
search()
```

#### Working on Windows  (Rtools)

Make sure to install RTool on windows platforms. It provides a tool chain to bring your windows environment more inline with unix and provides better functionality for your windows R environment by offering such things as `make`, `sed`, `tar`,`gzip`, a `C/C++ compiler`. This is very important when you start dealing with packages that need to be compiled and aren't provided already in binary form for windows. 

1. Install R and RStudio
2. Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/)  
3. RStudio run the following command:

```R
install.packages("devtools")
library(devtools) 
find_rtools() #should return "TRUE"
```

------

## Week 3

### Types of Analysis

**Descriptive**
Describe set of data, descriptions cannot be generalized without additional statistical modeling.*Description* and *interpretation* are different steps what we see why we see it

**Exploratory**
Find relationships we didn't know about.
NOTE: Correlation does not imply causation

**Inferential**
Using small sample of data to say something about a bigger population

- Stoical models
- Consider: 
  - estimating the quantity we care about
  - uncertainty about our estimate
- Heavy dependency on population and sample

**Predictive**
To use the data on some objects to predict values of another 

- $x \rightarrow y$ is not $y \rightarrow x$
- Accuracy heavily dependent on measuring the correct variables
- rule of thumb: more data and simple model 

**Causal Analysis**
Find out what happens to one variable when you change another variable

- Randomized studies are required to identify causation
  - Alternatives for non-random studies - complicated, sensitive to assumptions
- Causal relationships are identified as avg effects 
  - May not apply to every individual

**Mechanistic**
Understand the exact changes in variables that lead to changes in other variabels for indivudial objects

- Modeled by deterministic set of equations
- Random component of data = measure error
- Equation know, parameters not
  - Parameters inferred from data analysis  

### What is data?

> Data are values of <u>qualitative</u> or <u>quantitative</u> ==variables==, belonging to a set of items

Qualitative: Not ordered, think of it as a property
Quantitative: Measured on a continues scale, ordering on the scale

### Big Data

> *Big data* is high volume, high velocity, and/or high variety information assets that require new forms of processing to enable enhanced decision making, insight discovery and process optimization

Hadoop is not always the answer
Even with large data, it doesn't guarantee the answer to our question

### Experimental Design

Why is it important? It is the make or brake factor of the experiment.

Best practices for data sharing can be found on this [github repo]([https://github.com/jtleek/datasharing](https://github.com/jtleek/datasharing)).

**Data and code sharing:**

Code shared on [github](www.github.com)

Large sharing data can be done on [figshare](http://figshare.com).

#### Design Plan

1) Formulate our question in advance

*Example* Does colour of image used on website affect number of people who donate?

**Image from Obama campaign**

![statistical inference](.\Images/statistical inference.PNG)

- Some probability argument used to generate a sample population.
- Descriptive statistics state number of those who donated or not.
- Inferential statistics used to determine if finding of sample population will play out the same way with the general population.

**Confounding**

Misinterpreting correlation

***Example*** - We think S affects L when in fact A affects both S and L which both increase in tandem. 

![confounding](.\Images/confounding.PNG)

**Preventing some confounders**

- Fixing variables
- Stratify a sample: divide population into homogeneous groups before sampling
- if both two points aren't possible: *randomize

**Prediction vs Inference**
We need wide distribution of the two populations to tell what population influences the variables.

> Using data to *predict* an event that has yet to occur is statistical prediction.
>
> *Inferring* the value of a population quantity such as the average income of a country or the proportion of eligible voters who say they will vote ‘yes’ is statistical inference.
>
> Prediction and inference answer different types of statistical questions.

***Prediction key quantities:***

![Confusion Matrix](.\Images/Capture.PNG)

**Beware data dredging**
Continuing and modifying your hypothesis till the results match your statement

### Summary 

- Good Experiments 
  - Have replication
  - Measure variability
  - Generalize the problem we care about
  - Transparency
- Prediction is not inference
  - but both are important
- Beware data dredging


