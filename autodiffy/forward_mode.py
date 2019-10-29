import numpy as np
import operator


class ForwardMode:

    def __init__(self, val, der_dict):
        self.val = val
        self.der_dict = der_dict

    @staticmethod
    def combine_dicts(d1, d2, op):
        assert d1.keys() == d2.keys()
        return {var: op(d1[var], d2[var]) for var in d1.keys()}

    @staticmethod
    def product_rule(fm1, fm2):
        assert fm1.der_dict.keys() == fm2.der_dict.keys()
        return {var: fm1.val * fm2.der_dict[var] + fm2.val * fm1.der_dict[var] for var in fm1.der_dict.keys()}

    @staticmethod
    def quotient_rule(fm1, fm2):
        assert fm1.der_dict.keys() == fm2.der_dict.keys()
        return {var: (fm2.val * fm1.der_dict[var] - fm1.val * fm2.der_dict[var]) / (fm2.val ** 2)
                for var in fm1.der_dict.keys()}

    def __neg__(self):
        der_dict = {variable: -der for variable, der in self.der_dict.items()}
        return ForwardMode(-self.val, der_dict)

    def __add__(self, value):
        try:
            der_dict = ForwardMode.combine_dicts(self.der_dict, value.der_dict, operator.add)
            return ForwardMode(self.val + value.val, der_dict)

        except AttributeError:
            return ForwardMode(self.val + value, self.der_dict)

    def __radd__(self, value):
        return self + value

    def __sub__(self, value):
        try:
            der_dict = ForwardMode.combine_dicts(self.der_dict, value.der_dict, operator.sub)
            return ForwardMode(self.val - value.val, der_dict)

        except AttributeError:
            return ForwardMode(self.val - value, self.der_dict)

    def __rsub__(self, value):
        return ForwardMode(value - self.val, self.der_dict)

    def __mul__(self, factor):
        try:
            der_dict = ForwardMode.product_rule(self, factor)
            return ForwardMode(self.val * factor.val, der_dict)

        except AttributeError:
            der_dict = {var: factor * der for var, der in self.der_dict.items()}
            return ForwardMode(factor * self.val, der_dict)

    def __rmul__(self, factor):
        return self * factor

    def __truediv__(self, factor):
        try:
            der_dict = ForwardMode.quotient_rule(self, factor)
            return ForwardMode(self.val / factor.val, der_dict)

        except AttributeError:
            der_dict = {var: der / factor for var, der in self.der_dict.items()}
            return ForwardMode(self.val / factor, der_dict)

    def __rtruediv__(self, factor):
        der_dict = {var: (-factor * der) / (self.val ** 2) for var, der in self.der_dict.items()}
        return ForwardMode(factor / self.val, der_dict)

    def __pow__(self, value):
        try:
            der_dict = {var: self.val ** value.val * np.log(self.val) for var, der in self.der_dict.items()}
            return ForwardMode(self.val ** value.val, der_dict)

        except AttributeError:
            der_dict = {var: value * self.val ** (value - 1) * der for var, der in self.der_dict.items()}
            return ForwardMode(self.val ** value, der_dict)

    def __rpow__(self, value):
        der_dict = {var: value ** self.val * np.log(value) for var, der in self.der_dict.items()}
        return ForwardMode(value ** self.val, der_dict)

    def sin(self):
        der_dict = {var: np.cos(self.val) * der for var, der in self.der_dict.items()}
        return ForwardMode(np.sin(self.val), der_dict)

    def cos(self):
        der_dict = {var: -np.sin(self.val) * der for var, der in self.der_dict.items()}
        return ForwardMode(np.cos(self.val), der_dict)

    def tan(self):
        der_dict = {var: der / (np.cos(self.val) ** 2) for var, der in self.der_dict.items()}
        return ForwardMode(np.tan(self.val), der_dict)

    def exp(self):
        der_dict = {var: np.exp(self.val) * der for var, der in self.der_dict.items()}
        return ForwardMode(np.exp(self.val), der_dict)

    def log(self):
        der_dict = {var: der / self.val for var, der in self.der_dict.items()}
        return ForwardMode(np.log(self.val), der_dict)

    def log10(self):
        der_dict = {var: der / (self.val * np.log10) for var, der in self.der_dict.items()}
        return ForwardMode(np.log10(self.val), der_dict)

    def sqrt(self):
        return self ** 0.5
