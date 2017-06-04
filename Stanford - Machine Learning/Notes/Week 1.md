# Week 1

[TOC]

## Scope of Machine Learning

* Database mining 
* Applications that can't be implemented 
* Self-customizing programs 
* Understand human learning 

##### ***Definition***: Machine Learning

A computer program is said to *learn* from *Experience* $E$ with some *Task* $T$ and some performance measure $P$ if its performance on $T$, measured by $P$, improves with *Experience* $E$

### Types of Machines Learning

1. Supervised Learning
2. Unsupervised Learning
3. Reinforced Learning
4. Recommender systems

## Supervised Learning

##### ***Definition:*** Regression Problem

Predict a continues value

##### ***Definition:*** Classification Problem

Predict discrete value

##### Question: Infinite Number of Features?

How doe we make computers handle this? 
-> ==Support Vector Machine== (dealt with later)

## Unsupervised Learning

"Here's some data, let the machine find structure in the data by it self"

----- IMAGE

***Algorithm Example:*** Clustering

Clusters data according to similar properties it defines by it self

------ IMAGAE

##### ***Application Example:*** Cocktail Problem

----- IMAGE

By clustering identity by itself the 2 separate frequencies -> we can isolate the individual speakers

This can be done with the following code:

```octave
[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x);
## Where svd -> single value devision
```

## Linear Regression 

***Properties:*** 

* Supervised Learning
* Regression Problem

##### Notation:

$$
m = \text{training examples} \\
x = \text{input var / feature} \\
y = \text{output var / target} \\
(x,y) - \text{one training example} \\
(x^{(i)},y^{(i)}) - \text{ith training example}
$$

##### Structure:

-- image 



## Cost Function

## Intuition 