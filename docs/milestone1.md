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

### Prompt

Discuss how you plan on organizing your software package.

- What will the directory structure look like?
- What modules do you plan on including? What is their basic functionality?
- Where will your test suite live? Will you use `TravisCI`? `CodeCov`?
- How will you distribute your package (e.g. `PyPI`)?
- How will you package your software? Will you use a framework? If so, which one and why? If not, why not?
- Other considerations?

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

*ParseTree* class:

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

*ForwardMode* class:


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
- numpy
- operator module so that binary operators can be expressed as operator.xxxx(argument1, argument2)
- pythonds.trees for access to BinaryTree class
- pythonds.basic for access to Stack class in order to keep track of current node in binary tree











