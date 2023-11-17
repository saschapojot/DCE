import numpy as np
from scipy.special import jv
from scipy.special import hermite
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.misc import derivative
import math

nH=4
T=1
def H(x):
    return hermite(nH)(x)

def f(t):
    rst=(jv(-1/4,1/2*T)+T*jv(-5/4,1/2*T))*jv(1/4,1/2*T*(1+t/T)**2)\
        -(jv(1/4,1/2*T)-T*jv(5/4,1/2*T))*jv(-1/4,1/2*T*(1+t/T)**2)
    return rst

f0=f(0)

N=1000
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

#compute function R1
a=1

R1Vals=[a*(1+dVal**2)**(-1/2) for dVal in delta1Vals]

#compute function B1

B1Vals=[1/2*a**2*(1-1j*dVal)/(1+dVal**2) for dVal in delta1Vals]

#compute function D1

D1Vals=[]
for n in range(0,N+1):
    tn=tValsAll[n]
    dValn=delta1Vals[n]
    tmp=a*f0/f(tn)*((1+tn/T)*(1+dValn**2))**(-1/2)
    D1Vals.append(tmp)

#compute real part of function A1
ReA1=[1/2*D1Tmp**2 for D1Tmp in D1Vals]

#compute imaginary part of function A1

def fdot(t):
    """

    :param t:
    :return: derivative of f at t
    """

    val=derivative(f,t,dx=dt/50)
    return val


ImA1=[]
for n in range(0,N+1):
    tn=tValsAll[n]
    dValn=delta1Vals[n]
    D1Tmp=D1Vals[n]
    fTmp=f(tn)
    fdotTmp=fdot(tn)

    valTmp=-1/2*D1Tmp**2*dValn-fdotTmp/(2*fTmp)-1/(4*(T+tn))
    ImA1.append(valTmp)

#compute function A1
A1=[]
for n in range(0,N+1):
    A1.append(ReA1[n]+1j*ImA1[n])


L=10

M=2000

dx=2*L/M

xValsAll=[-L+m*dx for m in range(0,M+1)]

PsiValsAll=np.zeros((M+1,N+1),dtype=complex)

for m in range(0,M+1):#m is the index for position
    for n in range(0,N+1):#n is the index for time
        D1Tmp=D1Vals[n]
        xTmp=xValsAll[m]
        tTmp=tValsAll[n]
        d1Tmp=delta1Vals[n]
        A1Tmp=A1[n]
        PsiTmp=np.sqrt(D1Tmp)/(np.pi**(1/4)*np.sqrt(2**nH*math.factorial(nH)))*H(D1Tmp*xTmp)\
            *np.exp(-1j*(nH+1/2)*np.arctan(d1Tmp))*np.exp(-A1Tmp*xTmp**2)
        PsiValsAll[m,n]=PsiTmp

#normalization for each col
for n in range(0,N+1):
    colTmp=PsiValsAll[:,n]
    PsiValsAll[:,n]/=np.linalg.norm(colTmp,ord=2)


LHSMat=np.zeros((M+1,N+1),dtype=complex)
RHSMat=np.zeros((M+1,N+1),dtype=complex)

omegaValsAll=[(1+tn/T) for tn in tValsAll]

for m in range(1,M):
    for j in range(1,N):
        LHSMat[m,j]=1j*(PsiValsAll[m,j+1]-PsiValsAll[m,j-1])/(2*dt)

        omgTmp=omegaValsAll[j]

        RHSMat[m,j]=-1/2*(PsiValsAll[m+1,j]-2*PsiValsAll[m,j]+PsiValsAll[m-1,j])/dx**2\
                    +1/2*omgTmp**2*xValsAll[m]**2*PsiValsAll[m,j]


tdiff=[]
diff=[]
for n in range(1,N):
    tdiff.append(n*dt)
    tmp=np.linalg.norm(LHSMat[:,n]-RHSMat[:,n],ord=2)
    diff.append(tmp)


plt.figure()
plt.plot(tdiff,diff)
plt.savefig("diff.png")
plt.close()

