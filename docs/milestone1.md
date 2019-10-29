# Milestone 1 Document

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [1 Introduction](#1-introduction)
  - [1.1 What's a Derivative?](#11-whats-a-derivative)
  - [1.2 Why are Derivatives Important?](#12-why-are-derivatives-important)
  - [1.3 Symbolic Differentiation](#13-symbolic-differentiation)
  - [1.4 Finite Differences](#14-finite-differences)
  - [1.5 Advantages of Auto-Differentiation](#15-advantages-of-auto-differentiation)
- [2 Background](#2-background)
  - [2.1 Chain-Rule: The Core of Automatic Differentiation](#21-chain-rule-the-core-of-automatic-differentiation)
  - [2.2 Graph Structure of Calculations](#22-graph-structure-of-calculations)
  - [2.3 Dual Numbers](#23-dual-numbers)
- [How to Use `Autodiff`](#how-to-use-autodiff)
- [Software Organization](#software-organization)
  - [Directory Structure](#directory-structure)
  - [Software modules and basic functionality](#software-modules-and-basic-functionality)
  - [Software test suite](#software-test-suite)
  - [Software Distribution](#software-distribution)
- [Implementation](#implementation)
  - [Overview of classes:](#overview-of-classes)
    - [*ParseTree* class:](#parsetree-class)
    - [*ForwardMode* class:](#forwardmode-class)
  - [External dependencies:](#external-dependencies)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 1 Introduction

Differentiation forms the core of both traditional statistics methods such as Maximum-Likelihood Estimators, and leading edge methods in machine learning such as neural networks and sampling. A common problem in these fields is the need to take the gradient of an arbitrary function, which may not have a closed form analytical solution. 

Our package, `Autodiff`, addresses this by providing forward-mode automatic differentiation. Written in Python, it evaluates both the value of a user-supplied function and its derivative at a given point.

### 1.1 What's a Derivative?

In a single-variable sense, the derivative is the slope of the tangent line to the function at a given point. Formally, the derivative of a function, if it exists, is defined as

<p align="center"><img src="/docs/tex/a857ea8a2fe45ee07000b0d58378138e.svg?invert_in_darkmode&sanitize=true" align=middle width=202.83071819999998pt height=34.7253258pt/></p>

We can also view the derivative as the effect of an infinitesimal change in the value of <img src="/docs/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> on the value of the function <img src="/docs/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/>. This definition extends to multi-variable function.

### 1.2 Why are Derivatives Important?

Derivatives are useful in finding the maxima and minima of functions, and this has a variety of applications in statistics and machine learning.

**OLS \& Maximum Likelihood**

In statistics, we are often interested in discovering the 'right values' for parameters that explain a relationship between a response variable and some predictors. This is often phrased in terms of an optimization problem: what is the value of parameters that maximize some objective function, or equivalently, minimize some loss function.

Once phrased as an optimization problem, possibly subject to some constraints, a standard result from optimization suggests that we can examine the First-Order Condtions (FOCs) of the function by taking the derivative and setting that equal to zero in order to retrieve the optimal values.

Take, for example, the standard Ordinary Least Squares (OLS) estimator for a linear regression model of the form 

<p align="center"><img src="/docs/tex/c0bfae98689f9978243611c99749599c.svg?invert_in_darkmode&sanitize=true" align=middle width=98.30049735pt height=17.8466442pt/></p>

Suppose we want to find the estimate of <img src="/docs/tex/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode&sanitize=true" align=middle width=10.16555099999999pt height=22.831056599999986pt/> denoted <img src="/docs/tex/2012e6a4e80d4d65bda472f3676c43ec.svg?invert_in_darkmode&sanitize=true" align=middle width=10.562281949999988pt height=31.50689519999998pt/> that minimizes the Mean-Squared-Error (MSE) defined as <img src="/docs/tex/38a2b05691db864dc889394f60469ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=152.6500008pt height=27.6567522pt/>. We derive the First Order Conditions (FOCs) required to solve the optimization problem. Define

<p align="center"><img src="/docs/tex/938b772ad7623e0429e67ba8dc0565fb.svg?invert_in_darkmode&sanitize=true" align=middle width=199.97639969999997pt height=36.22493325pt/></p>

Since multiplication by a fixed scalar does not alter the result of an optimization problem, we can equivalently minimize <img src="/docs/tex/bc47eb54c896114e803bd8d0e087caf4.svg?invert_in_darkmode&sanitize=true" align=middle width=29.657642849999988pt height=24.65753399999998pt/>. Then we have

<p align="center"><img src="/docs/tex/ad53178074765a454c30d20928549b19.svg?invert_in_darkmode&sanitize=true" align=middle width=169.2744306pt height=34.7253258pt/></p> 

To find the MLE estimator <img src="/docs/tex/2012e6a4e80d4d65bda472f3676c43ec.svg?invert_in_darkmode&sanitize=true" align=middle width=10.562281949999988pt height=31.50689519999998pt/>, we set the FOC to zero.

<p align="center"><img src="/docs/tex/2478ae13185ebecd3527433596f9e26a.svg?invert_in_darkmode&sanitize=true" align=middle width=305.7906324pt height=19.8630366pt/></p>

However, another approach to estimation, known as **maximum-likelihood**, uses a different objective function. The core of maximum likelihood is to ask what is the parameter value that makes the data we do observe likely to occur. That is, given the data that we have observe, what is the most likely value of the parameter we want to estimate. This likelihood that we seek to maximize is nothing more than the probability density function of the distribution that we have assumed underlies the model we are estimating. We can equivalently maximize the log of the likelihood, also known as the log-likelihood. Since this likelihood can be any probability density function, an analytical solution for its derivative may not exist.

**Hamilton-Monte Carlo Sampling**

In Bayesian statistical analysis, we are often interested in sampling from the posterior distribution of a parameter under some assumption on the priors. In particular, unless we have chosen conjugate priors, the posterior distribution we want to sample from can be arbitrarily complex, or even only known up to some scalar proportion.

Under such circumstances, a popular sampling method to sample from an arbitrary distribution is known as **Hamilton-Monte Carlo**. Incorporating the notion of momentum from physics, it seeks to explore a wider space of possible values while still ensuring that the samples tend towards high-mass regions by making high-mass regions apply a 'gravity' of sorts on particles that are pushed off in random directions with random momentum.

The core of Hamilton-Monte Carlo relies on being able to take arbitrary gradients of the log of the target distribution we want to sample from. However, since the target distribution may be arbitrarily complex, working out its derivative by hand is not always possible.

### 1.3 Symbolic Differentiation

For simple functions such as exponentials, trigonometric functions, multiplication, and addition, it is possible to derive an analytic formula for the derivative. For example, we know that the derivative of <img src="/docs/tex/45d56c128bbdcd4414f84c155351f718.svg?invert_in_darkmode&sanitize=true" align=middle width=69.86299979999998pt height=26.76175259999998pt/> is given by <img src="/docs/tex/e2a1ce28d745dcfcc333b8e15e44dfa1.svg?invert_in_darkmode&sanitize=true" align=middle width=76.14153689999998pt height=24.7161288pt/>. Since we have the analytic formula for the derivative, we can compute it to machine precision.

However, this becomes harder to do when we want to take the derivative of, for example, functions defined in the computer science sense, algorithmically and employing flow control structures such as for loops. We can use expression trees, but these become unwieldy very quickly with increased function complexity. Furthermore, there are times where an analytic expression of the function is simply not possible. The transition from OLS estimation to maximum-likelihood in statistics that we saw in the previous section illustrates this.

Furthermore, it does not address the fundamental problem that taking the derivative of a function in this way is a highly bespoke problem. That is, every new function has to be worked through, and the process is highly artisanal, prone to human error.

### 1.4 Finite Differences

Given the way the derivative is defined, a natural approximation to the symbolic derivative where it does not exist in a closed-form analytical solution, or where it is difficult to work out, is to make a discrete analogue of the definition of the derivative. That is, choose some suitably small <img src="/docs/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/>, and approximate the derivative by computing

<p align="center"><img src="/docs/tex/b8378ec9689c31b225253b69936ef4b4.svg?invert_in_darkmode&sanitize=true" align=middle width=172.73743575pt height=34.7253258pt/></p>

The advantage of such an approach is that it applies to any arbitrary function, including functions in the 'computer science' sense defined possibly using for loops and other such constructions. It may also seem like a 'black-box' in the sense that it requires no artisanal derivations like symbolic differentiation does.

However, the trade-off with using any finite differences method is that we lose precision. That is, the accuracy of our approximation depends on choosing a small <img src="/docs/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/>, but there is a limit to how small a <img src="/docs/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> we can choose. Because the finite differences method relies on division, and division is an inherently expensive operation in terms of precision, we can only at best approximate the true derivative to several orders of magnitude lower than machine precision using the finite differences method.

### 1.5 Advantages of Auto-Differentiation

- Computes derivatives to machine precision.
- Does not rely on extensive mathematical derivations or expression trees, so it is easily applicable to a wide class of functions.

## 2 Background

### 2.1 Chain-Rule: The Core of Automatic Differentiation

The chain rule is a standard calculus result that allows us to take the derivative of nested functions. We use the notation that

<p align="center"><img src="/docs/tex/ca150cb4590264063d7f2ffeb7c7955f.svg?invert_in_darkmode&sanitize=true" align=middle width=131.08441499999998pt height=16.438356pt/></p>

Suppose we have a function <img src="/docs/tex/ffcbbb391bc04da2d07f7aef493d3e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=30.61077854999999pt height=24.65753399999998pt/> that is nested in another function <img src="/docs/tex/8824595f458085fe3bf467c4228300fc.svg?invert_in_darkmode&sanitize=true" align=middle width=27.169069949999987pt height=24.65753399999998pt/>, and we are interested in recovering the derivative of <img src="/docs/tex/8824595f458085fe3bf467c4228300fc.svg?invert_in_darkmode&sanitize=true" align=middle width=27.169069949999987pt height=24.65753399999998pt/> with respect to <img src="/docs/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>. The chain rule states that

<p align="center"><img src="/docs/tex/650bfcf42b304968dea6349d1635ed35.svg?invert_in_darkmode&sanitize=true" align=middle width=199.6345131pt height=37.0084374pt/></p>

In other words, if we can express a large, complex function as the nesting of smaller operations with known derivatives, then we can compute the derivative of the function at a particular value for each stage of this nesting structure and recover the overall derivative. These 'smaller operations' are known as **elementary operations**, with known derivatives. Our package will implement the following elementary operations:

- sin
- cos
- tan
- log
- log10
- exp
- sqrt

### 2.2 Graph Structure of Calculations

This approach naturally lends itself to writing out a complex function as the composite of elementary functions, where each node represents a function and the directed edges represent the computations that connect the nodes. As we go from the inputs to the final function, we perform the computations to evaluate the value of the function and its derivative for a given value.

This graph can be represented in a table form with the evaluation trace as well.

**Worked Example**

To illustrate how the graph structure works, we consider the following function

<p align="center"><img src="/docs/tex/d1a55cfc46629cc52752a4c88293812a.svg?invert_in_darkmode&sanitize=true" align=middle width=159.17237655pt height=18.312383099999998pt/></p>

evaluated at <img src="/docs/tex/a635d6f213f21ac52da2e25938d9c156.svg?invert_in_darkmode&sanitize=true" align=middle width=113.65085324999997pt height=27.94539330000001pt/>.

The evaluation trace looks like the following.

| Trace   | Elementary Function      | Current Value           | Elementary Function Derivative       | <img src="/docs/tex/644271f822c0a2a59907b3c936120ed7.svg?invert_in_darkmode&sanitize=true" align=middle width=21.153044549999994pt height=22.465723500000017pt/> Value  | <img src="/docs/tex/120888ade278ad9af7bffb67ec5c439f.svg?invert_in_darkmode&sanitize=true" align=middle width=20.77827839999999pt height=22.465723500000017pt/> Value  |
| :---: | :-----------------: | :-----------: | :----------------------------: | :-----------------:  | :-----------------: |
| <img src="/docs/tex/7e0dded6496a3ed3f3c0db74604087ac.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/7e0dded6496a3ed3f3c0db74604087ac.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/76c5792347bb90ef71cfbace628572cf.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/74370f5631ffcc0d659be47a6e26ea60.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=21.95701200000001pt/> | <img src="/docs/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> |
| <img src="/docs/tex/345508ce4e933b712fe803f442f74d63.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/345508ce4e933b712fe803f442f74d63.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/4eb105c60f67ef131323b9c0969450b8.svg?invert_in_darkmode&sanitize=true" align=middle width=8.099960549999997pt height=22.853275500000024pt/> | <img src="/docs/tex/017c3766de1bf45f78530c90083b6d31.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=21.95701200000001pt/> | <img src="/docs/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> |
| <img src="/docs/tex/0c2e0f3c3c97f7a8819ced883e72c6a6.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/fa874e46af4d176274d7c013050639f0.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/> | <img src="/docs/tex/ecf4fe2774fd9244b4fd56f7e76dc882.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/710ff052e0d410414878fc4898bcfb2b.svg?invert_in_darkmode&sanitize=true" align=middle width=40.93616999999999pt height=21.95701200000001pt/> | <img src="/docs/tex/ecf4fe2774fd9244b4fd56f7e76dc882.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> |
| <img src="/docs/tex/7dabc4eaf45b5547de66d633c8335c2f.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/2e55604b5e6e1a07e5e5236207a207e5.svg?invert_in_darkmode&sanitize=true" align=middle width=24.16674479999999pt height=21.18721440000001pt/> | <img src="/docs/tex/06798cd2c8dafc8ea4b2e78028094f67.svg?invert_in_darkmode&sanitize=true" align=middle width=8.099960549999997pt height=22.853275500000024pt/> | <img src="/docs/tex/69d49f642354e61ca119b76fa58df9a4.svg?invert_in_darkmode&sanitize=true" align=middle width=24.16674479999999pt height=21.95701200000001pt/> | <img src="/docs/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/76c5792347bb90ef71cfbace628572cf.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> |
| <img src="/docs/tex/84cb3de83fdfb7f77afb904200ae726b.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/2078c05d3eaa9257c0217fa28ee90615.svg?invert_in_darkmode&sanitize=true" align=middle width=51.56403284999998pt height=24.65753399999998pt/> | <img src="/docs/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/d71d062df2f736ed0da0740c66445de4.svg?invert_in_darkmode&sanitize=true" align=middle width=81.21019664999999pt height=24.65753399999998pt/> | <img src="/docs/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/f956e4ba3abf630e9642346ed4f9706b.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/> |
| <img src="/docs/tex/8b613d40166f51cc1dcd283bfebd2bdc.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=14.15524440000002pt/> | <img src="/docs/tex/70346ceb4309e739898f0b27fe9d9f39.svg?invert_in_darkmode&sanitize=true" align=middle width=52.808174099999995pt height=19.1781018pt/> | <img src="/docs/tex/ecf4fe2774fd9244b4fd56f7e76dc882.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/52e91f6b0cecb6d7fd1cd16ea557d31f.svg?invert_in_darkmode&sanitize=true" align=middle width=52.80815264999999pt height=21.95701200000001pt/> | <img src="/docs/tex/ecf4fe2774fd9244b4fd56f7e76dc882.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> | <img src="/docs/tex/f956e4ba3abf630e9642346ed4f9706b.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/> |

Based on the evaluation trace above, when <img src="/docs/tex/a635d6f213f21ac52da2e25938d9c156.svg?invert_in_darkmode&sanitize=true" align=middle width=113.65085324999997pt height=27.94539330000001pt/>, we have

<p align="center"><img src="/docs/tex/fc4db459ddfc4a8bb79e0b61cde67188.svg?invert_in_darkmode&sanitize=true" align=middle width=97.23844844999999pt height=30.1801401pt/></p>

<p align="center"><img src="/docs/tex/dbc32f4fdfc67fc35fe3b84617246453.svg?invert_in_darkmode&sanitize=true" align=middle width=114.1133466pt height=33.81208709999999pt/></p>

<p align="center"><img src="/docs/tex/5e737eb4e276f445dfe3303bde8cd911.svg?invert_in_darkmode&sanitize=true" align=middle width=126.89878079999998pt height=37.0084374pt/></p>

The evaluation trace can be visualized using the following graph.

![Computational Graph](./m1_image/bg_graph.png)

### 2.3 Dual Numbers

Dual numbers are a mathematical construct similar in structure to complex numbers. Dual numbers have a real and dual part, so we can write the dual number <img src="/docs/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> as

<p align="center"><img src="/docs/tex/007073b6000030cd13e94e70f3194bb8.svg?invert_in_darkmode&sanitize=true" align=middle width=80.5801128pt height=16.3763325pt/></p>

where <img src="/docs/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> is the real part and <img src="/docs/tex/15f93b25ba881e5829e8fc647b680fb2.svg?invert_in_darkmode&sanitize=true" align=middle width=12.43916849999999pt height=24.7161288pt/> is the dual part of the dual number. The number <img src="/docs/tex/9ae7733dac2b7b4470696ed36239b676.svg?invert_in_darkmode&sanitize=true" align=middle width=7.66550399999999pt height=14.15524440000002pt/> is a constant with the special property that <img src="/docs/tex/ad6370c8c8de22b67ebb85cbc747ef57.svg?invert_in_darkmode&sanitize=true" align=middle width=45.17680365pt height=26.76175259999998pt/>. Dual numbers have the following properties.

- Conjugate: The conjugate of <img src="/docs/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/>, denoted <img src="/docs/tex/baa7cc2f36af87229745595791fda227.svg?invert_in_darkmode&sanitize=true" align=middle width=16.55260694999999pt height=22.831056599999986pt/>, is defined as

<p align="center"><img src="/docs/tex/42e4c24bdc5ef2d675290bbcd9ab906f.svg?invert_in_darkmode&sanitize=true" align=middle width=88.1372019pt height=16.3763325pt/></p>

- Norm (or magnitude): The magnitude of <img src="/docs/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/>, denoted <img src="/docs/tex/96131d32ec473dc893b70065aa499ba2.svg?invert_in_darkmode&sanitize=true" align=middle width=74.61197204999999pt height=26.76175259999998pt/>, is defined as

<p align="center"><img src="/docs/tex/ac18a53f1fa3d1c7256381bdc9e26368.svg?invert_in_darkmode&sanitize=true" align=middle width=210.26616599999997pt height=18.312383099999998pt/></p>

- Polar form: The polar form of <img src="/docs/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is written

<p align="center"><img src="/docs/tex/e0b801a17494cdbf8215e5075a7e5141.svg?invert_in_darkmode&sanitize=true" align=middle width=120.50710815pt height=39.452455349999994pt/></p>

Salient to the question of automatic differentiation, dual numbers can be used to compute the derivatives of arbitrary functions. Suppose we have an arbitrary function <img src="/docs/tex/ffcbbb391bc04da2d07f7aef493d3e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=30.61077854999999pt height=24.65753399999998pt/> that we want to take the derivative of. We can compute <img src="/docs/tex/92f07220eff922f7e9b89e3eb33b73ab.svg?invert_in_darkmode&sanitize=true" align=middle width=66.27089864999998pt height=24.65753399999998pt/>, and the real part of that computation corresponds to the value of the function at the point <img src="/docs/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>, while the dual part of that computation corresponds to the value of the derivative of <img src="/docs/tex/ffcbbb391bc04da2d07f7aef493d3e2a.svg?invert_in_darkmode&sanitize=true" align=middle width=30.61077854999999pt height=24.65753399999998pt/> at the point <img src="/docs/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>.

**Worked Example**

We can take the example of <img src="/docs/tex/03b0f6bb834ba1e41336128850fd7ab4.svg?invert_in_darkmode&sanitize=true" align=middle width=109.57177769999997pt height=26.76175259999998pt/>. Since this is a simple function, we can take its derivative symbolically.

<p align="center"><img src="/docs/tex/da33f5fecc039c01d21a403ac08f810a.svg?invert_in_darkmode&sanitize=true" align=middle width=103.06488225pt height=17.2895712pt/></p>

Now we use dual numbers to reach the same conclusion.

<p align="center"><img src="/docs/tex/af76f82cc2530bcfec53c78bbf508f83.svg?invert_in_darkmode&sanitize=true" align=middle width=643.9938697499999pt height=18.312383099999998pt/></p>

Now we use the property that <img src="/docs/tex/ad6370c8c8de22b67ebb85cbc747ef57.svg?invert_in_darkmode&sanitize=true" align=middle width=45.17680365pt height=26.76175259999998pt/> and collect the terms to get

<p align="center"><img src="/docs/tex/22608f3efa3aeef755dd32728ac5e632.svg?invert_in_darkmode&sanitize=true" align=middle width=241.38656024999997pt height=18.312383099999998pt/></p>

We can see that the real part of this is the value of the function at the point <img src="/docs/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>, and the dual part of this is the value of the derivative at the point <img src="/docs/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>. We do not use dual numbers directly in our package, but in the same vein, we compute and return both the value of the derivative and the value of the function at a given point.

## How to Use `Autodiff`

This package will be primarily used by software developers and students looking to perform forward-mode differentiation on an expression at a given point. 

The implementation will focus on the command line interface between the user and the program. To start, the user shall create and then activate a virtual environment with `conda create -n [EXAMPLE_ENVIRONMENT] python=3.7 anaconda` and `source activate [EXAMPLE_ENVIRONMENT]`. 

After the environment is created, the user can then install our package with conda or pip install. 

When running the exectuable file the user will be prompted via terminal, after a welcome introduction, to choose from options (1), (2), or (3). Option (1) would specify the users intent to use a univariate function, (2) - a multivariate function, and (3) - a vector function. After choosing an option the user would be asked to specify variable character(s) they'd like to use to define their function expression. 

An example entry for case (1) would be `x`, as a primary varaiable character. `x` would then be stored in our *ForwardMode* class as the default character for the expression. 

The user would then be prompted, again via terminal, to enter their function directly into the terminal as close to as its written. Our program would parse this input string and look to build the expression fully in our *ForwardMode* object by searching for fundamental mathemtical operations such as +, -, *, /, ^, sin, cos,exponents, ln, log, etc and some of the other ways they could be written (i.e ^ == ** , etc). Once entered, our program would re-display their entry, but this time in our *ForwardMode*'s version to ensure accuracy of intent by the user, with them being able to select (1) if everything appears correctly or (2) if they need to re-enter the expression.

Lastly, after confirmation that the expression was interpreted by *ForwardMode* correctly, the user will be asked to specify the value(s) at which the expression should be evaluated at for each input variable they entered earlier.

*ForwardMode* then would calculate the value of the function and its derivative at the point values specificed by the user and output them in the terminal.

The input/output UI will be encapsulated in a `while` loop such that the user can continue to define and evaluate more functions, or quit when done.

## Software Organization

### Directory Structure 

```
Autodiff
│   README.md
|   LICENCSE.md
│   .travis.yaml
|   setup.py
|   .gitignore	
└─── Autodiff
│   │   __init__.py
│   │   forward_mode.py
│   │   binary_tree.py
│   │   interface.py
│   │   junk.py
│   │   parse_expression.py
│   │   parse_tree.py
│   │   reverse_mode.py
│   │   terms.py
│   └─── test
│       │   test.py
```

### Software modules and basic functionality

- Interface class: The GUI interface for our package 

- ForwardMode class: Takes in a scalar input and a function. Then computes the derivative of the function evaluated at the scalar input by using automatic differentiation. Stores the expression values and the derivatives

### Software test suite

- The test suite will will be placed in the test directory. `Travis CI` will be used for continous integration testing. `Codecov` will be used for code coverage testing.

### Software Distribution 

- Our package will be distributed on `PyPI`. 

## Implementation

1. Create a class called *ForwardMode* that keeps track of expression value and derivative(s) and updates these values as operations are applied to a *ForwardMode* object. The constructor takes in the value of the expression (e.g 2.0) and a dictionary that contains the partial derivatives of all the distinct variables in the expression (e.g. {x: 1, y: 0}) and sets this value and dictionary to be attributes. Then, a method will be created for each binary and unary operator that can be applied to a *ForwardMode* object. Unary methods include: *\_\_neg\_\_*, *sin*, *cos*, *tan*, *log*, *log10*, *exp* and *sqrt*. This means that when a *ForwardMode* object is passed into a numpy function such as np.sin, the *sin* method with be called instead of np.sin. All these methods will do two things: first, they'll update the value attribute of the *ForwardMode* object, and second they'll update the partial derivatives of the dictionary attribute. As a concrete example, the *sin* method will return a new *ForwardMode* object where the value is np.sin(self.value) and the partial derivative dictionary is *{variable: np.cos(self.value) \* derivative for variable, derivative in self.derivative_dict.items()}*. Meanwhile, binary methods include: *__add__*, *\_\_radd\_\_*, *\_\_sub\_\_*, *\_\_rsub\_\_*, *\_\_mul\_\_*, *\_\_rmul\_\_*, *\_\_truediv\_\_*, *\_\_rtruediv\_\_*, *\_\_pow\_\_* and *\_\_rpow\_\_*. For these methods, duck typing will be used to determine whether the other argument is a *ForwardMode* object or a numerical object (i.e. integer or float). As a concrete example, the *\_\_add\_\_* , method will first try to add the *value* attributes of the given *ForwardMode* object and the object passed in and will try to add their partial derivatives for each variable. However, if this results in an AttributeError (i.e. if the object passed in is not a *ForwardMode* object), then the value of the new *ForwardMode* object will be the *value* attribute of the given *ForwardMode* object plus the object passed in, and the dictionary of partial derivatives will remain the same. Because addition is a commutative operation, the *\_\_radd\_\_* method will simply return *self + argument*.

2.	Next, an expression that is a function of an arbitrary number of variables will be passed in (by the user) and broken into a list of its constituent parts. For example, *'exp(-(sin(x) – cos(y)) \*\* 2)'* will be processed and transformed to *['(', 'exp', '(', '-', '(', 'sin', '(', 'x', ')', '-', 'cos', '(', 'y', ')', ')', '\*\*', '2', ')', ')']*.

3. Next, the list that was generated in step 2 will be converted into a binary tree. This will be achieved by iterating over the elements of the list in order and following the following rules:
	* Beginning of list -> create tree with empty root node
	* Left parentheses ('(') -> insert left child with node value left blank and move down to this new left child
	* Right perentheses (')') -> move up tree to parent node
	* Binary operator ('+', '-', '\*', '/', '\*\*') -> set node value to given binary operator and insert right child with node value left blank. Then move down to this new right child 
	* Unary operator ('exp', 'sin', 'cos', 'tan', 'log', 'log10', 'sqrt', 'neg') -> set node value to given unary operator and insert left child with node value left blank (this subtree will not have a right child). Then move down to this new left child
	* Numeric value or variable (e.g. x, y) -> set node value to given number or variable (literally 'x' or 'y', etc.) and move up to parent node

4. Implement the forward mode by performing the following steps on the binary tree created in step 3 (refer to this function as *evaluate_tree*):
	* get the left and right children of the root node
	* if there is no left or right child this means that the give node is a leaf, so this node value should be evaluated: a numeric node value simply evaluates to that number, while a node value that is a variable (i.e. 'x' or 'y') is evaluated by creating a *ForwardMode* object where the *value* attribute is set to the value at which the given variable is to be evaluated and the dictionary of partial derivatives is created such that the value corresponding to all the variable keys is 0 except for the value corresponding to the node variable, which is set to 1 (i.e. if there are 3 variables, 'x', 'y' and 'z', and the node value is 'y', then the following partial derivative dictionary is created: {'x': 0, 'y': 1, 'z': 0}).
	* if there is no right child (but there is a left child), then this means that the value stored at the given node is a unary operator. Evaluating this node amounts to applying the given unary operator to the *ForwardMode* object that results from recursively applying this function (*evaluate_tree*) to the left tree of the given node. 
	* if there is both a right child and a left child, then this means that the value stored at the given note is a binary operator. Evaluating this node amounts to applying the given binary operator to the *ForwardMode* objects that result from recursively applying this function (*evaluate_tree*) to both the left tree and the right tree of the given node. 

### Overview of classes:

#### *ParseTree* class:

attributes: 

- *expression* (in form of list created in step 1)
- *variables* dictionary that maps all variables in expression to the value at which they are to be evaluated
- *binary_tree* that is initially set to None but will later be replaced with the binary tree created in step 3. 
- *result* that is initially set to None but will later be replaced with a ForwardMode object that contains the final value and all the partial derivatives 

methods:

- *build_parse_tree* that implements step 3 and sets binary_tree attribute to the resulting tree 
- *implement_forward_mode* that implements step 4 and sets result attribute to the ForwardMode object outputted from the binary tree
- *get_value* that retrieves evaluation of expression stored in *result* attribute
- *get_derivative* that retrieves dictionary of partial derivatives stored in *result* attribute

#### *ForwardMode* class:

attributes:

- *value*
- *derivative_dictionary*

methods:
- *sin*
- *cos*
- *tan*
- *log*
- *log10*
- *exp*
- *sqrt*
- *\_\_neg\_\_*
- *\_\_add\_\_*
- *\_\_radd\_\_*
- *\_\_sub\_\_*
- *\_\_rsub\_\_*
- *\_\_mul\_\_*
- *\_\_rmul\_\_*
- *\_\_truediv\_\_*
- *\_\_rtruediv\_\_*
- *\_\_pow\_\_* 
- *\_\_rpow\_\_*

### External dependencies: 
- `numpy`
- `operator` module so that binary operators can be expressed as operator.xxxx(argument1, argument2)
- `pythonds.trees` for access to `BinaryTree` class
- `pythonds.basic` for access to `Stack` class in order to keep track of current node in binary tree
