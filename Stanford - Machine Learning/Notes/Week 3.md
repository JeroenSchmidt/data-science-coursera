---
typora-copy-images-to: Images\Week3
---

# Week 3

[TOC]

## Classification & Representation

##### ***Terminology*** - for binary classification

1) $y \in (0,1)$ 

* 0 - *Negative Class*
  * also denoted $-$
* 1 - Positive Class
  * also denoted $+$

2) $y^{(i)}$ - label $i$

3)  $x^{(i)}$ - feature $i$

 

##### ***Example*** - Context

Using <u>Linear Regression</u> for classification problems

![1497251493005](Images/Week3/1497251493005.png)

In order to make prediction, we can then ***Threshold*** the values at $0.5$ for the regression line.

* if $h_\theta(x) \geq 0.5 \rightarrow y=1$ 
* if $h_\theta(x) \leq 0.5 \rightarrow y=0$ 

==NOTE== What if we have an outlier?

![1497251655970](Images/Week3/1497251655970.png)

Notice how our threshold of $0.5$ moves as a result of the outlier. 
==i.e. Applying linear regression to classification isnt a good idea as it can quickly become skewed==

The hypothesis $h_\theta(x)$ can be also be > 1 or < 0 even if the labels are only meant to be y = 1,0 

## Logistic Regression Model

### Hypothesis Function

We want $0\leq h_\theta(x)\leq 1$

Recall that $h_\theta(x)=\theta^Tx$, we want it to map with respect to the bounds specified so we apply a sigmoid function $g$ onto it
$$
h_\theta(x) = g(\theta^Tx)
$$
Where $g(z)$ is the ***Sigmoid Function*** (also called the *logistic function*)
$$
g(z) = \frac{1}{1+e^{-z}}
$$
![1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function](Images/Week3/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png)

##### ***Equation*** - Logistic Regression Hypothesis

$$
h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}
$$

#### Interpretation

$h_\theta(x) = $ estimated probability that $y=1$ on input x

*Expression:*

> Probability that $y=1$, given $x$, parametrized by $\theta$
>
> $h_\theta(x)=P(y=1| x;\theta)$ 

i.e. 

* Inverse, probability of $y=0$ occurring
  * $h_\theta(x) - 1 = P(y=0| x;\theta)$
* The summation is equal to 1 
  * $P(y=1| x;\theta) + P(y=0| x;\theta) = 1$

### Decision Boundary

When is $y=1$ w.r.t $\theta^Tx$ ? 

![1497255329567](Images/Week3/1497255329567.png)

Hence, we get $y=1$ when:
$$
h_\theta(x) = g(\theta^Tx) \geq 0.5 \\
\text{when } \theta^Tx \geq 0
$$
Conversely, $y=0$ when: 
$$
h_\theta(x)=g(z) < 0.5\\
\text{when }z = \theta^Tx < 0
$$

##### ***Example:*** linear boundary

Say we have the following data:

![1497255723579](Images/Week3/1497255723579.png)

Where the hypothesis has been determined to be: 
$$
h_\theta(x) = g(-3+ 1*x_1 + 1*x_2)
$$
Then $y=1$ if:
$$
-3+x_1+x_2 \geq 0\\
x_1 + x_2 \geq 3
$$
*Plotting this inequality:*
Everything to the right of the pink line is then determined to be the "red x" region

![1497255984444](Images/Week3/1497255984444.png)

##### ***Term:*** Decision Boundary

> * The pink line in the above diagram 
> * The function that separates the classification regions
>   * In the above example: $x_1+x_2=3$
> * ==NOTE== the decision boundary is a property of the hypothesis function and its parameters and not a property of the data set

***Example:*** Non-Linear decision boundaries

Say we have the following data:

![1497256807272](Images/Week3/1497256807272.png)

Where the hypothesis has been determined to be: 
$$
h_\theta(x) = g(-1+x_1^2+x_2^2)
$$
Then $y=1$ if:
$$
-1+x_1^2+x_2^2 \geq 0\\
x_1^2+x_2^2\geq 1
$$
*Plotting this inequality:*
Everything outside the pink circle is $y=1$

![1497257032106](Images/Week3/1497257032106.png)

with the decision boundary defined by $x_1^2+x_2^2=1$

### Cost Function

How do we choose our parameters $\theta$?

*System Set up*

![1497258531369](Images/Week3/1497258531369.png)



#### Context:

Recall that the *cost function* for linear regression is:
$$
J(\theta) = \frac{1}{m} \sum^m_{i=1} \frac{1}{2}\left(h_\theta(x)-y\right)^2
$$


But recall that the hypothesis for logistic regression is: (non-linear)
$$
h_\theta(x)=\frac{1}{1+e^{\theta^Tx}}
$$
This non-linearity will produce a "non-convec" cost function when put into the cost function $J(\theta)$

![1497262270214](Images/Week3/1497262270214.png)

The problem with this is that we will struggle to find the global optimal solution to the cost function.
==Hence== we define a cost function that will give us a convex function which is easier to optimize. 

#### ***Function:*** Logistic Regression cost Function

$$
J(\theta)=\frac{1}{m}\sum_m^{i=1} Cost(h_\theta(x^{(i)}),y^{(i)})\\
\text{Cost}(h_\theta(x),y)=\begin{cases}
    -log(h_\theta(x)), & \text{if $y=1$}.\\
    -log(1-h_\theta(x)), & \text{if } y=0.
  \end{cases}
$$

##### ***Intuition:*** $y=1$

Plot of $-log(h_\theta(x))$ as  $J(θ)$ vs $h_θ(x)$:

![Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class](Images/Week3/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png)

Observe that:
$$
\lim_{h_\theta(x) \to 1} Cost = 0 \\
\lim_{h_\theta(x) \to 0} Cost = \infty
$$
i.e The closer the hypothesis function $h$ predicts $0$ when it should be $1$ the greater the cost value $J$ will be (rapid growth towards infinity closer to 0 we get)

***Intuition:*** $y=0$

Similar to the previous example we have the following but the inverse  

Plot of $-log(1-h_\theta(x))$ as  $J(θ)$ vs $h_θ(x)$:

![Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class](Images/Week3/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png)

 

### Simplified Cost Function

$$
\text{Cost}(h_\theta(x),y)=\begin{cases}
    -log(h_\theta(x)), & \text{if $y=1$}.\\
    -log(1-h_\theta(x)), & \text{if } y=0.
  \end{cases}
$$

We can simplify the function above into:
$$
Cost(h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
$$

#### ***Function:*** Logistic Regression cost Function

$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

==NOTE== 

* The above cost function is convex
* This cost function can be derived from the principle of *maximum likelihood estimation* 

### Fitting Parameters - Gradient Descent 

#### ***Algorithm*** Gradient Descent

Minimizing $J(\theta)$
$$
\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}
$$

 #### Vectorized Implementation 

##### ***Vectorized*** - Cost Function

$$
h = g(X\theta)\\
g(z) = \frac{1}{1+e^{-z}} \\
J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right)
$$

##### ***Vectorized*** - Updating of Gradient Descent

$$
\theta := \theta - \alpha\frac{1}{m}\sum_{i=1}^{m}[(h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}]
$$

OR
$$
\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})
$$

### Advanced Optimization 



## Multiclass Classification

## Dealing with Over-fitting