
class ForwardMode:

    def __init__(self, val, der=1):
        self.val = val
        self.der = der

    def __neg__(self):
        return ForwardMode(-self.val, -self.der)

    def __add__(self, value):
        try:
            return ForwardMode(self.val + value.val, self.der + value.der)

        except AttributeError:
            return ForwardMode(self.val + value, self.der)

    def __radd__(self, value):
        return self + value

    def __sub__(self, value):
        try:
            return ForwardMode(self.val - value.val, self.der - value.der)

        except AttributeError:
            return ForwardMode(self.val - value, self.der)

    def __rsub__(self, value):
        return self - value

    def __mul__(self, factor):
        try:
            return ForwardMode(self.val * factor.val,
                               self.val * factor.der + self.der * factor.val)

        except AttributeError:
            return ForwardMode(factor * self.val, factor * self.der)

    def __rmul__(self, factor):
        return self * factor

    def __truediv__(self, factor):
        try:
            return ForwardMode(self.val / factor.val,
                               (factor.val * self.der - self.val * factor.der) / factor.val ** 2)

        except AttributeError:
            return ForwardMode(self.val / factor, self.der / factor)

    def __rtruediv__(self, factor):
        return ForwardMode(factor / self.val, (-factor * self.der) / self.val ** 2)

    def __pow__(self, exponent):
        try:
            return ForwardMode(self.val ** exponent.val,
                               exponent.val * self.val ** (exponent.val - 1) * exponent.der * self.der)

        except AttributeError:
            return ForwardMode(self.val ** exponent,
                               exponent * self.val ** (exponent - 1) * self.der)

    def sin(self):
        return ForwardMode(np.sin(self.val),
                           np.cos(self.val) * self.der)

    def cos(self):
        return ForwardMode(np.cos(self.val),
                           -np.sin(self.val) * self.der)

    def exp(self):
        return ForwardMode(np.exp(self.val), np.exp(self.val) * self.der)

    def log(self):
        return ForwardMode(np.log(self.val), self.der / self.val)

    def sqrt(self):
        return self ** 0.5


if __name__ == 'main':

    #expression = re.sub(, expr, 'ForwardMode(' + eval_point + ')')


    f1 = ForwardMode(4, 2)
    f2 = ForwardMode(5, 3)

    #f3 = f1 * f2
    #print(f3.val, f3.der)

    f3 = operator.truediv(f1, f2)
    print(f3.val, f3.der)


class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        self.leftChild = BinaryTree(newNode)

    def insertRight(self, newNode):
        self.rightChild = BinaryTree(newNode)

    def insertLeftExistingTree(self, tree):
        self.leftChild = tree

    def insertRightExistingTree(self, tree):
        self.rightChild = tree

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

# remove empty nodes due to extraneous parentheses
def remove_empty_nodes(tree):
    # check to see if we're traversed the whole tree
    if tree:
        # left child exists but has no stored value or operator
        if tree.getLeftChild() and not tree.getLeftChild().getRootVal():
            child = tree.getLeftChild()
            while True:
                # check if empty node is a leaf
                if not child.getLeftChild():
                    raise ValueError('Empty node is leaf')

                if child.getLeftChild().getRootVal():
                    tree.replaceLeft(child.getLeftChild())
                    break
                child = child.getLeftChild()

        remove_empty_nodes(tree.getLeftChild())
        remove_empty_nodes(tree.getRightChild())

#variables = {'x': np.pi/16, 'y': np.pi/3}
#expression = 'exp(-(sin(x)-cos(y)) ** 2)'
#expression = '(x - exp(-2((sin(4*x)) ** 2)))'
#expression = '((np.pi/16) - exp((sin((4*(np.pi/16))) ** 2)))'
#expression = '((np.pi/16) - exp(-2(sin((4*(np.pi/16))) ** 2)))'
#expression = '(x - exp(-2((sin(4*x)) ** 2)))'
#expression = 'exp((sin((4*(np.pi/16))) ** 2))'
#expression = '((x*y)+sin(x))'
#expression = '(x + y + sin(x*y)'