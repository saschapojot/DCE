import numpy as np
from scipy.special import jv
from scipy.special import hermite
from scipy.integrate import quad
import matplotlib.pyplot as plt


nH=4
T=1
def H(x):
    return hermite(nH)(x)

def f(t):
    rst=(jv(-1/4,1/2*T)+T*jv(-5/4,1/2*T))*jv(1/4,1/2*T*(1+t/T)**2)\
        -(jv(1/4,1/2*T)-T*jv(5/4,1/2*T))*jv(-1/4,1/2*T*(1+t/T)**2)
    return rst

f0=f(0)

N=500
dt=T/N
tValsAll=[dt*n for n in range(0,N+1)]

#compute the integral in delta1 from ndt to (n+1)dt


def intDelta(n):
    """

    :param n: starting index of time
    :return: the integral in delta1 from ndt to (n+1)dt
    """
    tn=n*dt
    tnp1=(n+1)*dt
    val,_=quad(lambda tau:f0**2/((tau+T)*f(tau)**2),tn,tnp1,limit=10000)
    return T*val


def deltaIntegrand(tau):
    return f0**2/((tau+T)*f(tau)**2)



# funcValsAll=[deltaIntegrand(tau) for tau in tValsAll]
# plt.figure()
# plt.plot(tValsAll,funcValsAll)
# plt.savefig("tmp.png")
#
# plt.close()
Deltadelta1IntVals=[intDelta(n) for n in range(0,N)]

delta1Vals=[0]+list(np.cumsum(Deltadelta1IntVals))

