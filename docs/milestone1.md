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

```
autodiff
│   README.md
│   .travis.yaml
|   .gitignore	
|   LICENCSE
└─── implementation
│   │   binary_tree.py
│   │   forward_mode.py
│   │   interface.py
│   │   junk.py
│   │   parse_expression.py
│   │   parse_tree.py
│   │   reverse_mode.py
│   │   terms.py
│   └─── test
│       │   test.py
```
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
