import numpy as np
import operator


class Terms:

    def __init__(self):

        self.binops = {'+', '*', '/', '**'}

        self.binops2operator = {'+': operator.add,
                                '-': operator.sub,
                                '*': operator.mul,
                                '/': operator.truediv,
                                '**': operator.pow}

        self.unop2operator = {'sin': np.sin,
                              'cos': np.cos,
                              'tan': np.tan,
                              'exp': np.exp,
                              'log': np.log,
                              'log10': np.log10,
                              'sqrt': np.sqrt,
                              'neg': operator.neg}
