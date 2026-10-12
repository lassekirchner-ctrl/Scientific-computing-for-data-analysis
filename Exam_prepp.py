#tenta 2023/10/31
import numpy as np

C = np.array([4.00, 2.25, 1.45, 1.00, 0.65, 0.25, 0.06])
r = np.array([0.398, 0.298, 0.238, 0.198, 0.158, 0.098, 0.048])

#linearize variables
ln_C = np.log(C)
ln_r = np.log(r)

#create A using vander, equiavlent to columns stacking [ones, ln(c)]

A = np.vander(ln_C, N = 2, increasing= True)
b = ln_r

ATA = A.T @ A
ATb = A.T @ b

c0, c1 = np.linalg.solve(ATA, ATb)

k = np.exp(c0)
n = c1

print(k, n)