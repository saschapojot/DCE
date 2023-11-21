import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import math
from scipy.special import hermite
from datetime import datetime
from scipy.linalg import block_diag


j1H=1
j2H=1

def H(n,x):
    """

    :param n: order of Hermite polynomial
    :param x:
    :return: value of polynomial at x
    """
    return hermite(n)(x)


g0=1

omegac=100
omegam=3
omegap=2.95
er=50
lmd=(er-1/er)/(er+1/er)*(omegam-omegap)
theta=np.pi/7
Deltam=omegam-omegap

tTot=0.1
N1=2
N2=5
L1=5
L2=5

dx1=2*L1/N1
dx2=2*L2/N2

dtEst=0.002
M=int(tTot/dtEst)
dt=tTot/M

x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])

#construct H0
H0=(-1/2*omegac-1/2*Deltam)*np.eye(N1*N2,dtype=complex)

D2=np.diag(x2ValsAll)

S2=np.diag(np.power(x2ValsAll,2))

allD2=[D2 for n1 in range(0,N1)]
allS2=[S2 for n1 in range(0,N1)]
blockD2=block_diag(*allD2)
blockS2=block_diag(*allS2)

def H1Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H1 at time step j
    """
    tj=j*dt
    return -1/2*g0*np.sqrt(2*omegam)*np.cos(omegap*(tj+1/2*dt))*blockD2

#construct H2
IN2=np.eye(N2,dtype=complex)
blockH2=[IN2*x1ValsAll[n1]**2 for n1 in range(0,N1)]
diagH2=block_diag(*blockH2)
H2=1/2*omegac**2*diagH2


#construct H3
H3=(1/2*lmd*omegam*np.cos(theta)+1/2*Deltam*omegam)*block_diag(*blockS2)


blockH4=[D2*x1ValsAll[n1]**2 for n1 in range(0,N1)]
diagH4=block_diag(*blockH4)

def H4Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H4 at time step j
    """

    tj=j*dt

    return g0*omegac*np.sqrt(2*omegam)*np.cos(omegap*(tj+1/2*dt))*diagH4


P2Ones=np.ones(N2-1)
P2=np.diag(P2Ones,k=1)+np.diag(-P2Ones,k=-1)

blockH5=[P2 for n1 in range(0,N1)]
blockA5IN2=[IN2 for n1 in range(0,N1)]

diagH5=block_diag(*blockH5)
diagA5IN2=block_diag(*blockA5IN2)

def H5Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H5 at time step j
    """

    tj=j*dt

    A5j=-1j*g0*omegac*np.sqrt(2/omegam)*np.sin(omegap*(tj+1/2*dt))*diagH2\
        +1j*1/2*g0*np.sqrt(2/omegam)*np.sin(omegap*(tj+1/2*dt))*diagA5IN2

    return A5j*1/(2*dx2)*diagH5



