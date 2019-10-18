from terms import Terms


class ParseExpression:

    def __init__(self):
        self.terms = Terms()

    def _get_function(self, expr, var_names):
        for idx, char in enumerate(expr):
            if not char.isalpha():
                if char.isnumeric():
                    raise ValueError('invalid expression')
                candidate_expr = expr[:idx]
                if candidate_expr in self.terms.unop2operator or candidate_expr in var_names:
                    return idx, candidate_expr
                raise ValueError('invalid expression')

        raise ValueError('invalid expression')

    @staticmethod
    def _get_number(expr):
        for idx, digit in enumerate(expr):
            if not digit.isnumeric():
                if digit.isalpha():
                    raise ValueError('invalid expression')
                return idx, expr[:idx]

        return expr

    @staticmethod
    def _get_numpy_expr(expr):
        for idx, char in enumerate(expr):
            if not char.isalpha():
                if char.isnumeric():
                    raise ValueError('invalid expression')
                return idx, expr[:idx]

    def parse_expression(self, expr, var_names):
        tokens = []
        idx = 0
        while idx < len(expr):

            if expr[idx: idx + len('np.')] == 'np.':
                expr_len, np_expr = ParseExpression._get_numpy_expr(expr[idx + 3:])
                tokens.append(np_expr)
                idx += (len('np.') + expr_len)

            elif expr[idx].isalpha():
                expr_len, fun = self._get_function(expr[idx:], var_names)
                tokens.append(fun)
                idx += expr_len

            elif expr[idx].isnumeric():
                expr_len, num = ParseExpression._get_number(expr[idx:])
                tokens.append(num)
                idx += expr_len

            elif expr[idx: idx + len('**')] == '**':
                tokens.append('**')
                idx += len('**')

            elif expr[idx] == ' ':
                idx += 1

            else:
                tokens.append(expr[idx])
                idx += 1

        print(tokens)
        return tokens

