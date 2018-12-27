import numpy as np

a = range(1000)

b1 = a[0:10]
b2 = a[10:20]

c1 = []
c1.append(b1)
c1.append(b2)

c2 = []
c2.append(b1)
c2.append(b2)

d = []
d.append(c1)
d.append(c2)
d = np.array(d)
print(np.shape(d))

import keras


