#tenta 2023/10/31
import numpy as np
import scipy 

def Q4_2023():
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
#Q4_2023()

#tenta 2022
def Q3():

    #Q, R = np.linalg.qr(A)

    Q = np.array([[-0.5, 0.5, -0.5],
                  [-0.5, -0.5, 0.5],
                  [-0.5, -0.5, -0.5], 
                  [-0.5, 0.5, 0.5]])

    R = np.array([[-2, -3, -2],
                  [0, -5, 2],
                  [0, 0, -4]])

    y = np.array([[2],
                 [-2],
                 [8],
                 [-4]])

    QT = Q.T
    QR = Q @ R

    sol = np.linalg.solve(R, QT @ y)
    print(sol)
    #sol, residuals, rank, s = np.linalg.lstsq(QR, y, rcond = None)
    ##or 
    full_sol = np.linalg.lstsq(QR, y)
    print((2 + 3*1.6)/(-2))
    return print(full_sol[0])

#Q3()


def Q3_gemini():
    Q = 0.5* np.array([[1, 1, -1],
                       [1, -1, 1],
                       [1, -1, -1],
                       [1, 1, 1]])

    Q_trans = Q.T
    #print(Q_trans)

    R = np.array([[3, 1, 2],
                  [0, -2, 4],
                  [0, 0, 2]])

    y = np.array([[4],
                  [0],
                  [-2],
                  [2]])

    A = Q @ R


    # print(Q)
    # print(R)
    # print(y)

    full_sol = np.linalg.lstsq(A, y)

    sols = full_sol[0]
    x1, x2, x3 = np.round(sols[0], 2), np.round(sols[1], 2), np.round(sols[2], 2)
    return print(x1, x2, x3)
#Q3_gemini()

import random
def f(x):
    return 1 / (1+0.5*x**2)

def Q4_2022(a, b, n):
    total = 0
    rands = []
    for _ in range (0, n):
        rand = a + random.random()
        rands.append(rand)
        total += f(rand)
    #print(f'Random numbers are: \n {rands}.')
    return (b-a) * (total/n)

#print(Q4_2022(1, 2, 5))


def Q5_2022(m ): #m = number of sims 
    means = np.zeros(m)
    #print(means )
    for i in range (m):
        means[i] = (Q4_2022(1, 2, 5))

    std = np.std(means)
    mean = np.mean(means)
    print(f'The mean of {m} simulations is {mean}.')
    print(f'The std of the mean is: {std}')
    return means, std

#Q5_2022(1000000)

#print(420/ 45.1)


#problem solving
##3A. 
import math
def g(x):
    return (x**2)*math.e**x

def problem_3A(a, b, n):
    total = 0
    for _ in range (0, n):
        rand = np.random.uniform(1, 3)
        total += g(rand)
    return (b-a) * (total/n)


#print(problem_3A(1, 3, 10**6))

def problem_3b(a, b, n):
    monte_value = problem_3A(a, b, n)
    exact = 97.7094028
    return print(f'The absolute error is: {monte_value - exact} for n = {n}') 

#problem_3b(1, 3, 10**6)

def prob_3c(m):
    print('The values should follow a noral distribution.')
    print(f'If m = {m}:')
    results = np.zeros(m)
    for i in range (m):
        results[i] = problem_3A(3, 1, 10**6)
    std = np.std(results)
    error = 1.96 * std / (math.sqrt(m))
    return print(f'The error is: {error}') 

#prob_3c(160) #takes like 5 if m = 100 mins! 

'''prob_3c(100) #takes like 5 mins!:
If m = 100:
The error is: 0.01989448177243998

If m = 10:
The error is: 0.05261450246846799

print(0.05261450246846799/0.01989448177243998) = 2.6446782112894933
'''


#exam 2022
#exc 7 
def exc7():
    age = np.array([52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72])
    weight = np.array([78, 83, 88, 96, 89, 100, 85, 85, 94, 76, 98])
    bp = np.array([132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166])


    A = np.column_stack([np.ones(len(age)), age, weight])
    B = bp
    Q, R = np.linalg.qr(A)

    c_qr = np.linalg.solve(R, Q.T @ B)
    print(c_qr)


def exc3_2024():
    total = 224 + 171 + 329+120+156 ##1000
    freqs = [224, 171, 329, 120, 156]
    for i in range(len(freqs)):
        print(f'Frequency for {i +1} is {round(freqs[i] / total, 2)}')

#exc3_2024()
from math import e 
from math import log
def Q4_2024():
    t = np.array([1, 2, 3], float)
    b = np.array([[log(135)], [log(62)], [log(35)]], float)
    A = np.column_stack([np.ones_like(t), t])
    #print(A, b)
    a = np.linalg.lstsq(A, b) 
    coeffs = a[0]
    a0, a1 = coeffs[0], coeffs[1]
    # print(a0)
    # print(a1)
    def y (x):
        return e**(a0) * e**((x)*a1)

    for i in range (1, 10):

        print(f'Week: {i}, revenue gross: {y(i)}')
    

#Q4_2024()

def Q5_2025():
    print(f'Error for M = 1000: {1.96 * 0.33 / math.sqrt(1000)}')
    print(f'Error for M = 25000: {1.96 * 0.33 / math.sqrt(25000)}')
    print(f'Error factor by: {(1.96 * 0.33 / math.sqrt(25000) )/ ((1.96 * 0.33 / math.sqrt(1000)))}')
#Q5_2025()


def Q6_2025():
    lamda1 = -4
    lambda2 = 2

    print(f'Error shrinks by: (lambda2/lambda1)^k.')
    print(f'Error for 12 iterations is: {round(0.001*(lambda2/lamda1)**5, 8)}')
    ##=> implies answer 3*10**-5


#Q6_2025()
#print(3/0.2)

def Q8():
    A = np.array([[1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, 1], 
                  [-1, 1, 0],
                  [-1, 0, 1],
                  [0, -1, 1]])

    b = np.array([[1237],
                 [1941],
                 [2417],
                 [711],
                 [1177],
                 [475]])

    Q, R = np.linalg.qr(A)
    QT = Q.T

    print(Q @ QT)

    result = np.linalg.solve(R, QT @ b )
    #print(result)

#Q8()


def Q9A(m, n):
    N = 500
    v = np.ones(N)
    for i in range (N):
        A = np.random.rand(m, n)
        singular_vals = np.linalg.svd(A, compute_uv= False )
        v[i] = (singular_vals[7]) #sigma8
    std = v.std()
    error = 1.96 * std / math.sqrt(N)
    print(f'Distance is: {v.mean()}')
    print(f'Error is: {error} with 95% confidence interval.')

#Q9A(20, 15)


def Q9B(m, n):
    N = 500
    v = np.ones(N) #stores one value per simulation! 
    for i in range (N):
        A = np.random.rand(m, n)
        singular_vals = np.linalg.svd(A, compute_uv= False )
        v[i] = (np.sum(singular_vals[:3]**2 / np.sum(singular_vals**2))) #sigma0 to sigma2
    std = v.std()
    error = 1.96 * std / math.sqrt(N)
    print(f'Top 3 share is: {v.mean()}')
    print(f'Error is: {error} with 95% confidence interval.')

#Q9B(20, 15)


##re-do of exam 2023
from math import e
def regression_2023():
    b = np.array([[0.398],
                  [0.298],
                  [0.238],
                  [0.198],
                  [0.158],
                  [0.098],
                  [0.048]])

    b = np.log(b)

    A = ([[e, 4],
          [e, 2.25],
          [e, 1.45],
          [e, 1],
          [e, 0.65],
          [e, 0.25],
          [e, 0.06]])

    A = np.log(A)

    Q, R = np.linalg.qr(A)
    QT = Q.T
    

    result = np.linalg.solve(R, QT @ b)
    k = e**(result[0])
    c = e**(result[1])
    print(f'k = {k}, c = {c}')
    res_lsqt = np.linalg.lstsq(A, b)
    print(res_lsqt)


#regression_2023()


## 
def Q3_2025():
    R = np.array([[2, 0],
                  [0,2 ],
                  [0,0],
                  [0,0]])

    Q = 0.5* np.array([[-1, 1, 1, 1],
                       [1, -1, 1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])

    b = np.array([[2],
                 [0],
                 [-6],
                 [8]])

    R1 = np.array([[2, 6],
                  [0, 2]])
    

    Q1 = 0.5* np.array([[-1, 1],
                        [1, -1],
                        [1, 1],
                        [1, 1]])

    Q2 = 0.5* np.array([[1, 1],
                        [1, 1],
                        [-1, 1],
                        [1, -1]])
    
    
    QT = Q.T
    Q1T = Q1.T
    Q2T = Q2.T

    Q1Tb = Q1T @ b 
    Q2Tb = Q2T@b

    sol = np.linalg.solve(R1, Q1T @ b)
    print(sol)
    #print(Q2Tb)

#Q3_2025()

#print(32**(1/3))
def Q4_2024():
    X = np.array([[1, 0.1],
                  [1, 0.2],
                  [1, 0.3],
                  [1, 0.4],
                  [1, 0.5]])

    y = np.array([[0.1],
                  [0.2],
                  [0.3],
                  [0.4],
                  [0.5]])

    y = np.sin(y)

    XTX = X.T @ X

    XTy = X.T @ y

    print(XTX)
    print(XTy)

    sol = np.linalg.solve(XTX, XTy)


    print(sol)
#Q4_2024()

def Q9_2024(N):
    M = 100
    def helper(N, M = 1):
        x_vals = np.random.uniform(-1, 2, N)
        y_vals = np.random.uniform(0, math.pi, N)

        x_term = 3*x_vals*1
        y_term = math.pi * np.sin(y_vals)

        return x_term*y_term.mean()

        #print(np.shape(totals), totals[995:])


    totals = helper (N)
    

    total_total = np.mean([helper(N, M) for i in range (M)])

    return totals.mean(), total_total

#print(Q9_2024(10**4))