# Milestone 1 Document

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Introduction](#introduction)
- [Background](#background)
  - [Prompt](#prompt)
  - [Response](#response)
- [How to Use <PACKAGE NAME>](#how-to-use-package-name)
  - [Prompt](#prompt-1)
  - [Response](#response-1)
- [Software Organization](#software-organization)
  - [Prompt](#prompt-2)
  - [Response](#response-2)
- [Implementation](#implementation)
  - [Prompt](#prompt-3)
  - [Response](#response-3)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

Describe the problem the software solves and why it's important to solve that problem.

## Background

### Prompt

Describe (briefly) the mathematical background and concepts as you see fit. You do not need to give a treatise on automatic differentation or dual numbers. Just give the essential ideas (e.g. the chain rule, the graph structure of calculations, elementary functions, etc). Do not copy and paste any of the lecture notes. We will easily be able to tell if you did this as it does not show that you truly understand the problem at hand.

### Response

## How to Use <PACKAGE NAME>

### Prompt

How do you envision that a user will interact with your package? What should they import? How can they instantiate AD objects?

**Note: This section should be a mix of pseudo code and text.** It should not include any actual operations yet. Remember, you have not yet written any code at this point.

### Response

## Software Organization

- Directory Structure 
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
- Software modules and basic functionality
    - Interface class: The GUI interface for our package 
    - ForwardMode class: Takes in a scalar input and a function. Then computes the derivative of the function evaluated at the scalar input by using automatic differentiation. Stores the expression values and the derivatives

- Software test suite
    - The test suite will will be placed in the test directory. `Travis CI` will be used for continous integration testing. `Codecov` will be used for code coverage testing.

- Software Distribution 
    - Our package will be distributed on ``PyPI`. 

### Response

## Implementation

### Prompt

Discuss how you plan on implementing the forward mode of automatic differentiation.

- What are the core data structures?
- What classes will you implement?
- What method and name attributes will your classes have?
- What external dependencies will you rely on?
- How will you deal with elementary functions like `sin`, `sqrt`, `log`, and `exp` (and all the others)?
- Be sure to consider a variety of use cases. For example, don't limit your design to scalar functions of scalar values. Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors. Don't forget that people will want to use your library in algorithms like Newton's method (among others).

### Response
