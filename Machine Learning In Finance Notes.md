# Machine Learning In Finance Notes

*Kaiyi Li*

*NYU Shanghai Class of 2019*

*NYU Stern Class of 2020*

*kaiyi.li@stern.nyu.edu*

**This notebook is created by Kaiyi Li at 23:08 November 2nd (Beijing Time)  for the NYU Tandon Financial Engineering course track, including four courses: **

* Guided tour of Machine Learning in Finance
* Fundamentals of Machine Learning in Finance
* Reinforcement Learning in Finance
* Overview of Advanced Methods of Reinforcement Learning in Finance

[toc]

# Guided tour of Machine Learning in Finance

(Week 1, Lesson 3~4)

-------------------------------------------------

## Concepts

The landscaple of Machine learning includes 

* Perception Tasks
	* Supervised Learning
		* Regression: Give continous, return countinous
		* Classification: Give continous, return discrete
	* Unsupervised Learning
		* Clustering: Give continous, return discrete. **Difference between clustering and Classification: we don't have labels input **
		* Representation Learning: Learn Representer Function. Dimension reduction/ Feature reduction.
* Action Tasks
	* Reinforcement Learning
		* Optimization of strategy for a task: Learn policy function-- Pick optimal action to maximize total reward
		* IRL: Learn objectives from behavior: Learn Objectives from behavior-- Find the reward function that explains behavior 

## Applications in Finance

* Regression: Most commonly used. Stock return prediction, Earning prediction, credit loss forecast, algorithmic trading
* Classification: rating prediction, default modeling, credit card fraud
* Clustering: Stock segmentation
* Reprentation Learning: Factor Modeling, De-noising, Regime Change Detection
* Reinforcement Learning (Optimization): Asset management, Trading Strategies
* Inverse Reinforcement Learning: consumer behavior, trading strategy etc.

### Euclidean Distance

Calculate Euclidean Distance in python

>Euclidean Distance:
>
>The straight line distance.
>$$
>D(X,Y) = \sqrt{(\sum_{i=1}^{n}|X_i-Y_i|^2)}
>$$
>
>
>Manhattan Distance:
>
>The distance between two points in manhattan. 
>$$
>D(X,Y) = (\sum_{i=1}^{n}|X_i-Y_i|)
>$$



#### Code to calculate Euclidean Distance in python

```python
from scipy import spatial
import numpy
from sklearn.metrics.pairwise import euclidean_distances

import math

######################################################################### Calculating distance by using  python math function
############################################################# 2 D array
x1 = [1,1]
x2 = [2,9]
eudistance =math.sqrt(math.pow(x1[0]-x2[0],2) + math.pow(x1[1]-x2[1],2) )
print("eudistance Using math ", eudistance)

############################################################# 3 D array
# x1 = [1,1,4]
# x2 = [10,2,7]
# # Calculating distance by using math
# eudistance = math.sqrt(math.pow(x1[0]-x2[0],2) + math.pow(x1[1]-x2[1],2) + math.pow(x1[2]-x2[2],2) )
# print("eudistance Using math ", eudistance)

######################################################################### Calculating distance by using scipy
eudistance = spatial.distance.euclidean(x1, x2)
print("eudistance Using scipy", eudistance)


######################################################################### Calculating distance by using numpy
x1np=numpy.array(x1)
x2np=numpy.array(x2)
eudistance = numpy.sqrt(numpy.sum((x1np-x2np)**2))
print("eudistance Using numpy", eudistance)

eudistance = numpy.linalg.norm(x1np-x2np)
print("eudistance Using numpy", eudistance)


######################################################################### Calculating distance by using sklearn
eudistance =  euclidean_distances([x1np], [x2np]) # for some strange reasons, values needs be in 2-D array
print("eudistance Using sklearn", eudistance)

print('*** Program ended ***')
```

------------------------------------

## Generalization 

(Week2, Lesson 1~)

Consider a regression problem
$$
y = f(x) + \epsilon
$$
Here $\epsilon$ is an error.

* $E[\epsilon] = 0, E[\epsilon^2] = \sigma^2$

This produces $E[y] = f(x), Var(y) = \sigma^2$

### Bias variance decomposition

$$
E[(y-\hat{f}(x))^2] = bias^2+variance+noise
$$

We have
$$
bias = E[f-\hat{f}]\\
variance = Var(\hat f)\\
noise = Var(y) = \sigma^2
$$

### Bias Variance Trade-Off

For Complex models (more features), tend to have <u>low bias</u> and <u>high variance</u>

For Simple Models, (less features), tend to have <u>high bias</u> and <u>low variance</u>

>No Free Lunch Theorem
>
>There are no such Machine Learning Algorithm can be universally better than any other Machine Learning Algorithms

## Overfitting and Model Capacity

### Train-Test Split

![image-20191103005600468](/Users/likaiyi/Documents/python_programming_for_finance/MLFF/image-20191103005600468.png)

* Train Set is used for modeling, generalization.
* Test set is used to detect when a ML algorithm starts to overfit data.
* Over-fitting

![image-20191103001718239](/Users/likaiyi/Documents/python_programming_for_finance/MLFF/image-20191103001718239.png)

## Linear Regression

Given a dataset $(X,y)_{data}$

Perform a train-test Split, we then have

$[(X_{train},y_{train}),(X_{test},y_{test})]$ 

### Architecture

$$
\hat y = XW
$$

Where $X$ is a $N \times D$ design matrix, and $W \in R^D$ is a vector of parameters.

### Performance Measure

#### Mean Squared Error (MSE)

$$
MSE_{test} = \frac{1}{N_{test}} \sum_{n=1}^{N_{test}}(\hat y_i^{test}-y_i^{test})
$$

To find the optimal vector of W, the gradient of $MSE_{train} = 0$

### Optimal Gradients (For Linear Regression)

$$
\nabla_W MSE_{train} = \nabla_W \frac{1}{N_{train}} ||\hat Y^{train}-Y^{train} ||^2 = 0\\
\Rightarrow\\
\nabla_W(X^{train}W-y^{train})^T(X^{train}W-y^{train}) = 0\\
\Rightarrow\text{(take derivatives with resepect to W)}\\
2(X^TX)W-2(X^Ty) = 0\\
\Rightarrow\\
W = (X^TX)^{-1}X^Ty
$$

### Multicollinearity

>Multicollinearity could lead to instability in predictions.

### Regularization

The idea of regularization is to **modify the objective function of minimization of MSE on the train set** so that the MSE on the test set could have smaller variance.
$$
J(W) = MSE_{train}(W) + \lambda \Omega(W)
$$
Where we have

* $\lambda $ as a regularization parameter
* $\Omega$ as regularizer.

Regularization is to choose the best $\lambda $

#### Regularizer

##### LASSO regularization

$$
\Omega(W ) = W^TW =||W||_2\\
$$

Penalizes large weight.

##### Ridge regularization

$$
\Omega(W) = \sum_i |W_i| = ||W||_1
$$

Enforce a sparse solution

##### Entropy regularization

$$
\Omega(W) = \sum_iW_ilog(W_i)\\
(W_i\ge 0 ,\sum_i W_i = 1)
$$

Motivated by Bayesian statistics.

## Hyperparameters and Validation Set

**Hyperparameters** are any quantative features of ML models that are not directly optimized by minimizing in-sample loss such as MSE.

**They could control model capacity.**

Examples: degree of a polynomial regression, regularization parameter $ \lambda$, number of layers in networks, levels in tree.

#### How to choose hyperparameters

* Split a training set into training and validation sets
* Use cross-validation.

### Cross-Validation 

K-fold cross validation (usually K = 5 or K = 10)

![image-20191103005600468](/Users/likaiyi/Documents/python_programming_for_finance/MLFF/image-20191103005600468.png)

The best hyperparameter will be the one that provides the smallest average of the sample error.

## Intro to TensorFlow

(Week 3)

### Dataflow

![image-20191103020047019](/Users/likaiyi/Documents/python_programming_for_finance/MLFF/image-20191103020047019.png)

### Tensors:

Rank n= 0: a number

Rank n= 1: a list of numbers

Rank n= 2: a list of lists

Rank n= 3: a list of lists of lists

E.g: Stock data as a rank-3 tensor  (Date, Stock, Features)

**Tensors+Dataflow = TensorFlow**

### Reverse-mode Autodiff

Reverse-mode Autodiff implements automatic derivatives of any functions

* Forward pass (from imputs to outputs)
* Backward pass(from outputs to inputs)
* Use the chain rule for a composite function













