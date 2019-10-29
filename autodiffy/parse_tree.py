import numpy as np
from pythonds.basic import Stack

from binary_tree import BinaryTreeExtended
from forward_mode import ForwardMode
from interface import expression, variables
from parse_expression import ParseExpression
from reverse_mode import reverse_mode
from terms import Terms


class ParseTree:

    def __init__(self, expr, variables):
        self.expr = expr
        self.variables = variables
        self.terms = Terms()
        self.parse_expr = ParseExpression()
        self.parse_tree = None
        self.result = None
        self.storage_dict = {var: 0 for var in self.variables}

    def build_parse_tree(self):
        token_lst = self.parse_expr.parse_expression(self.expr, self.variables)
        stack = Stack()
        tree = BinaryTreeExtended('')
        stack.push(tree)
        current_tree = tree
        idx = 0
        while idx < len(token_lst):
            token = token_lst[idx]

            if token == '(':
                current_tree.insertLeft('')
                stack.push(current_tree)
                current_tree = current_tree.getLeftChild()
                idx += 1

            elif token in set(self.terms.binops) | {'-'}:

                # node already has value so tree needs to be extended
                if current_tree.getRootVal():
                    if stack.size() == 1:
                        tree = BinaryTreeExtended('')
                        tree.insertLeftExistingTree(current_tree)
                        current_tree = tree
                    else:
                        current_child = current_tree.getLeftChild()
                        new_child = BinaryTreeExtended('')
                        new_child.insertLeftExistingTree(current_child)
                        current_tree.insertLeftExistingTree(new_child)
                        stack.push(current_tree)
                        current_tree = current_tree.getLeftChild()

                if token in self.terms.binops:
                    current_tree.setRootVal(token)
                    current_tree.insertRight('')
                    stack.push(current_tree)
                    current_tree = current_tree.getRightChild()
                    idx += 1

                elif token == '-':

                    # '-' should be treated as a minus sign
                    if current_tree.getLeftChild():
                        current_tree.setRootVal(token)
                        current_tree.insertRight('')
                        stack.push(current_tree)
                        current_tree = current_tree.getRightChild()
                        idx += 1

                    # '-' should be treated as a negation
                    elif not token_lst[idx + 1].isdigit() and not token_lst[idx + 1] in self.variables:
                        current_tree.setRootVal('neg')
                        current_tree.insertLeft('')
                        stack.push(current_tree)
                        current_tree = current_tree.getLeftChild()
                        idx += 1

                    # '-' should be treated as a negative number
                    else:
                        current_tree.setRootVal('*')
                        current_tree.insertRight('')
                        stack.push(current_tree)
                        current_tree = current_tree.getRightChild()
                        current_tree.setRootVal('-' + token_lst[idx + 1])
                        idx += 2
                        parent = stack.pop()
                        current_tree = parent
                        current_tree.insertLeft('')
                        stack.push(current_tree)
                        current_tree = current_tree.getLeftChild()

                else:
                    raise ValueError

            elif token == ')':
                current_tree = stack.pop()
                idx += 1

            elif token in self.terms.unop2operator:
                current_tree.setRootVal(token)
                idx += 1

            elif token.isnumeric() or token in self.variables:
                current_tree.setRootVal(token)
                parent = stack.pop()
                current_tree = parent
                idx += 1

            else:
                raise ValueError

        self.parse_tree = tree

    def _traverse_tree(self, tree):
        if tree:
            print(tree.getRootVal())
            print(tree.getDerivatives())
            print()
            self._traverse_tree(tree.getLeftChild())
            self._traverse_tree(tree.getRightChild())

    def traverse(self):
        self._traverse_tree(self.parse_tree)

    def _evaluate_val(self, val, mode):
        if val in self.variables:
            if mode == 'forward':
                derivative_dict = {variable: 1 if variable == val else 0 for variable in self.variables}
                return ForwardMode(self.variables[val], derivative_dict)

            return self.variables[val]

        if val.replace('.', '', 1).replace('-', '', 1).isdigit() or val.startswith('np.'):
            return eval(val)

        raise ValueError('invalid leaf value (not numeric or variable)')

    def _evaluate_tree(self, tree, mode):
        left_tree = tree.getLeftChild()
        right_tree = tree.getRightChild()

        if mode == 'reverse' and tree.getVal():
            return tree.getVal()

        if not right_tree:

            # no children so must be leaf
            if not left_tree:

                evaluation = self._evaluate_val(tree.getRootVal(), mode)
                if mode == 'reverse':
                    tree.setVal(evaluation)
                return evaluation

            # no value stored in node
            if not tree.getRootVal():
                return self._evaluate_tree(left_tree, mode)

            if tree.getRootVal() not in self.terms.unop2operator:
                raise ValueError('invalid node value')

            evaluation = self.terms.unop2operator[tree.getRootVal()](self._evaluate_tree(left_tree, mode))
            if mode == 'reverse':
                tree.setVal(evaluation)
            return evaluation

        if tree.getRootVal() not in self.terms.binops2operator:
            raise ValueError('invalid node value')

        evaluation = self.terms.binops2operator[tree.getRootVal()](self._evaluate_tree(left_tree, mode),
                                                                   self._evaluate_tree(right_tree, mode))
        if mode == 'reverse':
            tree.setVal(evaluation)
        return evaluation

    def _forward_pass_helper(self, tree):

        if tree and tree.getLeftChild():
            left_tree = tree.getLeftChild()
            right_tree = tree.getRightChild()

            if not right_tree:

                # no value stored in node
                if not tree.getRootVal():
                    return self._forward_pass_helper(left_tree)

                if tree.getRootVal() not in self.terms.unop2operator:
                    raise ValueError('invalid node value')

                tree.insertDerivatives(reverse_mode(tree.getRootVal(), self._evaluate_tree(left_tree, 'reverse')))

                self._forward_pass_helper(left_tree)

            else:
                if tree.getRootVal() not in self.terms.binops2operator:
                    raise ValueError('invalid node value')

                tree.insertDerivatives(reverse_mode(tree.getRootVal(), self._evaluate_tree(left_tree, 'reverse'),
                                                    self._evaluate_tree(right_tree, 'reverse')))

                self._forward_pass_helper(left_tree)
                self._forward_pass_helper(right_tree)

    def _forward_pass(self):
        return self._forward_pass_helper(self.parse_tree)

    def _reverse_pass_helper(self, tree, num):
        root_val = tree.getRootVal()

        if not root_val:
            self._reverse_pass_helper(tree.getLeftChild(), num)

        elif root_val in self.variables:
            self.storage_dict[tree.getRootVal()] += num

        elif root_val in self.terms.unop2operator:
            self._reverse_pass_helper(tree.getLeftChild(), num * tree.getDerivatives()[0])

        elif root_val in self.terms.binops2operator:
            self._reverse_pass_helper(tree.getLeftChild(), num * tree.getDerivatives()[0])
            self._reverse_pass_helper(tree.getRightChild(), num * tree.getDerivatives()[1])

    def _reverse_pass(self):
        return self._reverse_pass_helper(self.parse_tree, 1)

    def _evaluate_function_value(self):
        return self._evaluate_tree(self.parse_tree, 'reverse')

    def implement_forward_mode(self):
        self.result = self._evaluate_tree(self.parse_tree, 'forward')

    def implement_reverse_mode(self):
        self._forward_pass()
        self._reverse_pass()
        self.result = ForwardMode(self._evaluate_function_value(), self.storage_dict)

    def get_value(self):
        return self.result.val

    def get_derivative(self, variable):
        return self.result.der_dict[variable]


parse_tree = ParseTree(expression, variables)
parse_tree.build_parse_tree()
parse_tree.implement_forward_mode()
print(parse_tree.get_value())
print(parse_tree.get_derivative('x'))
print(parse_tree.get_derivative('y'))
parse_tree.implement_reverse_mode()
print(parse_tree.get_value())
print(parse_tree.get_derivative('x'))
print(parse_tree.get_derivative('y'))


if False:
    expression = '(x + y - sin(x*y))'
    variables = {'x': 3, 'y': 2}

    #if __name__ == '__main__':

    #parse_tree.traverse()
    #print(parse_tree._evaluate_function_value())

    #parse_tree.implement_forward_mode()

