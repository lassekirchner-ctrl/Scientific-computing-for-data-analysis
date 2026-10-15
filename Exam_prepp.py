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
    
    QR = Q @ R
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

problem_3b(1, 3, 10**6)

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


''' if we run 3c with m = 160



'''

#exc4 
