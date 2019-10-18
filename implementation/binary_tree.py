from pythonds.trees import BinaryTree


class BinaryTreeExtended(BinaryTree):

    def __init__(self, rootObj):
        BinaryTree.__init__(self, rootObj)
        self.leftChildDer = None
        self.rightChildDer = None
        self.val = None

    def insertLeft(self, node):
        self.leftChild = BinaryTreeExtended(node)

    def insertRight(self, node):
        self.rightChild = BinaryTreeExtended(node)

    def insertLeftExistingTree(self, tree):
        self.leftChild = tree

    def insertRightExistingTree(self, tree):
        self.rightChild = tree

    def insertDerivatives(self, vals):
        self.leftChildDer = vals[0]
        self.rightChildDer = vals[1]

    def getDerivatives(self):
        return self.leftChildDer, self.rightChildDer

    def setVal(self, val):
        self.val = val

    def getVal(self):
        return self.val
