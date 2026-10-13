#==================================
# Gillespies algoritm
# prop - function returning propensities (1D numpy-array)
# stoch - function returning stoichiometry matrix (2D numpy-array)
# X0 - initial state (list, tuplet or numpy-array)
# tspan - simulation time interval
# coeff - model parameters
#==================================
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def SSA(prop, stoch, X0, tspan, coeff):
    # prop  - propensities, a function
    # stoch - stoichiometry vector, a function
    # X0 - Initial state vector
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1,np.size(X0)])
    Xarr[0,:] = X0
    t = tvec[0]
    X = X0
    sMat = stoch()
    while t<tspan[1]:
        r1, r2 = np.random.uniform(0,1, size=2)  # Find two random numbers on uniform distr.
        re = prop(X,coeff)
        cre = np.cumsum(re)
        a0 = cre[-1]
        if a0 < 1e-12:
            break

        tau = np.random.exponential(scale=1/a0) # Random number exponential distribution
        # Find which reaction
        cre=cre/a0
       # r=0
       # while cre[r]<r2:
       #     r+=1
        
        r = np.argwhere(cre>r2)[0][0]

        t+= tau
        # if new time is larger than final time, skip last calculation
        if t > tspan[1]:
            break
        
        tvec = np.append(tvec,t)
        X = X + sMat[r,:]
        Xarr = np.vstack([Xarr,X])

     # If iterations stopped before final time, add final time and no change
    if tvec[-1] < tspan[1]:
        tvec=np.append(tvec,tspan[1])
        Xarr=np.vstack([Xarr,X])          
            
    return tvec, Xarr

##own code
def seir_rhs(t, y, beta, alpha, gamma, N):
    S, E, I, R = y
    infection = beta * I * S / N
    # d shorthand for d prim
    dS = -infection
    dE = infection - alpha * E
    dI = alpha * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def simulate_ode(beta, alpha=1/5, gamma=1/10, N=100, I0=2, T=60):
    y0 = [N - I0, 0, I0, 0]          # S, E, I, R
    t_eval = np.linspace(0, T, 601)
    sol = solve_ivp(seir_rhs, (0, T), y0, t_eval=t_eval,
                    args=(beta, alpha, gamma, N), rtol=1e-8, atol=1e-10)
    return sol


def R0(beta, gamma, S0, N):
    return beta / gamma * S0 / N


def exc1():
    N, I0 = 100, 2
    alpha, gamma = 1/5, 1/10
    S0 = N - I0

    # --- Part 1: beta = 0.6 ---
    beta = 0.6
    print(f"beta = {beta}: R0 = {R0(beta, gamma, S0, N):.3f}")
    sol = simulate_ode(beta, alpha, gamma, N, I0)

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(["S", "E", "I", "R"]):
        plt.plot(sol.t, sol.y[i], label=name)
    plt.xlabel("Time (days)")
    plt.ylabel("Individuals (% of population)")
    plt.title(f"SEIR ODE model, transmission rate (beta) = {beta}")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- Part 2: reduce beta until R0 = 1 ---
    beta_crit = gamma * N / S0       # solves R0 = 1
    print(f"beta which yeilds R0 = 1 is: beta {beta_crit:.4f}")

    plt.figure(figsize=(8, 5))
    for b in [0.6, 0.3, 0.15, beta_crit, 0.08]:
        s = simulate_ode(b, alpha, gamma, N, I0)
        plt.plot(s.t, s.y[2], label=f"beta={b:.3f}, R0={R0(b, gamma, S0, N):.2f}")
    plt.xlabel("Time (days)")
    plt.ylabel("Infectious I, %")
    plt.title("Infectious vs time for decreasing transmission rate (beta)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


#exc1()

### end exc 1


## exc 2
def prop(X, coeff):
    """Propensities, in the same order as the rows of stoch()."""
    S, E, I, R = X
    beta, alpha, gamma, N = coeff
    return np.array([beta * S * I / N,   # S -> E  (infection)
                     alpha * E,          # E -> I  (becomes infectious)
                     gamma * I])         # I -> R  (recovery)
 
 
def stoch():
    """Stoichiometry matrix: one row per reaction, one column per state (S,E,I,R)."""
    return np.array([[-1,  1,  0, 0],
                     [ 0, -1,  1, 0],
                     [ 0,  0, -1, 1]])
 
 
def exc2():
    N, I0 = 100, 2
    beta, alpha, gamma = 0.6, 1/5, 1/10
    coeff = (beta, alpha, gamma, N)
    X0 = np.array([N - I0, 0, I0, 0])
    tspan = (0, 60)
 
    # ODE reference from task 1
    sol = simulate_ode(beta, alpha, gamma, N, I0, T=tspan[1])
 
    # One SSA run
    tvec, Xarr = SSA(prop, stoch, X0, tspan, coeff)
 
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    names = ["S", "E", "I", "R"]
    for i, name in enumerate(names):
        axs[0].plot(sol.t, sol.y[i], label=name)
        axs[1].step(tvec, Xarr[:, i], where="post", label=name)
    axs[0].set_title("Deterministic (ODE)")
    axs[1].set_title("Stochastic (one SSA run)")
    for ax in axs:
        ax.set_xlabel("Time (days)")
        ax.grid(alpha=0.3)
        ax.legend()
    axs[0].set_ylabel("Individuals")
    plt.tight_layout()
    plt.show()

#exc2()
#SSA(prop, stoch, X0, tspan, coeff):
# prop  - propensities, a function
# stoch - stoichiometry vector, a function
# X0 - Initial state vector


def exc3():
    N, I0 = 100, 2
    beta, alpha, gamma = 0.6, 1/5, 1/10
    coeff = (beta, alpha, gamma, N)
    X0 = np.array([N - I0, 0, I0, 0])
    tspan = (0, 60)
    num_sims = 30

    sol = simulate_ode(beta, alpha, gamma, N, I0, T=tspan[1])
    plt.figure(figsize=(10, 6))

    for i in range(num_sims):
        tvec, Xarr = SSA(prop, stoch, X0, tspan, coeff)
        plt.step(tvec, Xarr[:, 2], where="post", color="gray", alpha=0.4, label="SSA simultaions" if i == 0 else "")

    plt.plot(sol.t, sol.y[2], color="crimson", linewidth=2.5, label="ODE (Deterministic)")

    plt.xlabel("Time (days)")
    plt.ylabel("Infectious Individuals ($I$)")
    plt.title(f"Exercise 3: {num_sims} Stochastic SSA Trajectories vs. Deterministic ODE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

exc3()