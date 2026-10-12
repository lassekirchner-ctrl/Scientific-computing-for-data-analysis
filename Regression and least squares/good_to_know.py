import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial



p1 = Polynomial([-1, 0, 3, 2])
p2 = Polynomial([-2, 1, 3])

print(p1, '\n', p2)

#or linspace usage:
'''
n = np.linspace(1, 10, 10)
print(n)
'''



x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
plt.plot(x,p1(x))
plt.title('p1 over interval')
plt.xlabel('x')
plt.ylabel('p1')
plt.legend(['p1(x)'])
plt.show()
plt.figure(1)

y = np.arange (-2, 2, 0.4)
plt.plot(y,p2(y))
plt.title('p2 over y')
plt.xlabel('y')
plt.ylabel('p2')
plt.legend(['p2(y)'])
plt.show()
plt.figure(2)


### polynomial handling ###
linear_fit1 = Polynomial.fit(x, example_array[:, 0], deg=1, window=[1960,2020])
linear_fit2 = Polynomial.fit(x, example_array[:, 1], deg=1, window=[1960,2020])
quad_fit = Polynomial.fit(x, example_array[:, 1], deg=2, window=[1960,2020])
cubic_fit = Polynomial.fit(x, example_array[:, 1], deg=3, window=[1960,2020])








#IVP solver
#np.solve_ivp 


#understanding dimensions of diffrent datas:
z = np.array([[1, 2, 3]])
print('z = ', z)
print('Shape: ', z.shape) # note that the shape referes to number of column - 1, and number of rows - 3. 
print('Dimension: ',z.ndim)


#matrix creation:
A = np.array([[2, 4, -6], 
              [1, 5, 3],
              [1, 3, 2]])

B = np.array([[1, 3, 2], 
              [2, 4, 6]])

print(A)
print(B)

#create identity matrix
I = np.identity(4)
print(I)
#or use the eye command
C = np.eye((4))
print(C)

#join two vectors
z = np.zeros(16)
v = np.array([3, 1, 2, 0])
q = np.concatenate((v, z))
#print(q)


#flytta diagonalen i identitetsmatrisen
D = np.eye(6,k=-1)
#print(D)


#skapa vectorer med vissa intervall eller steg:
y = np.arange(1, 20)
#print(y)
x = np.arange(1, 20, 2)
#print(x)

##how to transpose matrices
'''
print(A)
print(np.transpose(A))
###
###
print(A)
print(A.T)'''


#solving a root finding equation:
from numpy.polynomial import Polynomial

p2 = Polynomial([-2, 1, 3])

# Take the derivative
dp2 = p2.deriv()

# Find where the derivative is zero
roots = dp2.roots()
print("Critical point(s):", roots)

# Evaluate p2 at that point to get the extreme value
extreme_x = roots[0]
extreme_y = p2(extreme_x)
print(f"Extremum at x = {extreme_x:.4f}, p2(x) = {extreme_y:.4f}")



