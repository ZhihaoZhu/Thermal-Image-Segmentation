import numpy as np

x = [[1,2,3],[4,5,6],[7,8,9]]
x = zip(*x[::-1])
print(x)