import numpy as np

expression = input('expression to differentiate:\n')
expression = '(' + expression + ')'

variables = {}

while True:
    new_var = input('variable name:\n')
    new_value = input('evaluate ' + new_var + ' at:\n')

    variables[new_var] = eval(new_value)

    val = int(input('\nEnter Option:\n1)Finished specifying variables\n2)Specify another variable\n'))
    if val == 1:
        break
