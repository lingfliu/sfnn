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


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.1)

y = sp.sin(x)

x_interp = np.arange(-10, 10, 0.1/3)
#
# y_interp = sp.interp(x_interp, x, y)
# plt.stem(x,y)
# plt.show()
# plt.stem(x_interp, y_interp)
# plt.show()


x = np.arange(0,3)
y = np.arange(0,3)
xy = np.dot(x,y)
print(x, y, xy)

print(sp.exp(1))

print(sp.sqrt(-1))
print(-1j*2)

print(np.zeros(4))


a = []
a.append([i for i in range(10)])
a.append([i*2 for i in range(10)])
a.append([i*3 for i in range(10)])


b = a[:][1]
print(b)

print(np.flipud(a))

print(np.sqrt(a))

ph0 = [i for i in range(3)]
ph = []
ph.append(0)
for v in ph0:
    ph.append(v)
ph.append(0)
for v in ph0[::-1]:
    ph.append(-v)

print(ph)

print(np.random.rand(10))
print(np.arange(-10,10))

print(np.ones(shape=(2,3)))

a = np.random.rand(10)*10
b = [m for m in filter(lambda x:a[x]<5, range(10))]

print(a,b)


t0 = 0
t1 = 2.6
print(np.linspace(t0, t1, 100))
print(np.arange(t0, t1, 0.1))

a = [1,2,3]
print(np.sum(a))
a = 8
print(np.remainder(a, 2*sp.pi))
# a = [i for i in range(30)]
# print(a[30//2::-1])
print(sp.log2(2))
import ecgsyn
(a,b) = ecgsyn.ecgsyn(sfecg=250, anoise=0, N=100)
# plt.plot(a[:1000])
# plt.plot(b[:1000])
# plt.show()


# import pickle
# pickle.dump((a,b), open('test.dat','wb'))
# del a,b
# (a,b) = pickle.load(open('test.dat','rb'))
# plt.plot(a[:5000])
# plt.plot(b[:5000])
# plt.show()

import datetime

print(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
