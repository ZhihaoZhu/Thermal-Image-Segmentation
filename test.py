import numpy as np
import collections

x = " a  b c "
x1 = " aa  b c "

y = collections.Counter(x)
y1 = collections.Counter(x1)


print(not y1-y)
