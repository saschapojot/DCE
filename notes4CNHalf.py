import numpy as np
from scipy.special import jv
from scipy.special import hermite
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.misc import derivative
import math
from datetime import datetime
from multiprocessing import Pool
# from multiprocessing import set_start_method

# set_start_method('forkserver')

#verify CN algorithm

#construct exact values of psi

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

tExactStart=datetime.now()
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


L=5

M=2000

dx=2*L/M

xValsAll=[-L+m*dx for m in range(0,M)]

PsiValsAll=np.zeros((M,N+1),dtype=complex)

for m in range(0,M):#m is the index for position
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
    PsiValsAll[:,n]/=np.linalg.norm(colTmp,ord=2)*np.sqrt(dx)

# for n in range(0,N+1):
#     print("0th elem: "+str(abs(PsiValsAll[0,n]))+", last elem: "+str(abs(PsiValsAll[-1,n])))
tExactEnd=datetime.now()
print("construct exact solution: ",tExactEnd-tExactStart)

P=np.zeros((M,M),dtype=complex)
for m in range(0,M):
    P[m,m]=-2
    P[m,(m+1)%M]=1
    P[(m-1)%M,m]=1

P/=-2*dx**2


def omega(t):
    return 1+t/T

def eigj(j):
    """

    :param j: time step j
    :return: HDj, eigenvalues and eigenvectors of HDj
    """
    Vj=np.zeros((M,M),dtype=complex)
    for m in range(0,M):
        Vj[m,m]=xValsAll[m]**2
    Vj*=1/2*omega(j*dt+1/2*dt)**2

    HDj=P+Vj

    vals,vecs=np.linalg.eigh(HDj)

    return [j,HDj,vals,vecs]

tStepsAll=[j for j in range(0,N)]

tEigStart=datetime.now()


procNum=12

pool0=Pool(procNum)

retAll=pool0.map(eigj,tStepsAll)

tEigEnd=datetime.now()

print("eig time: ",tEigEnd-tEigStart)

sortedRetAll=sorted(retAll,key=lambda item: item[0])


def oneStepEvolution(j,psij):
    """

    :param j: current step
    :param psij: current wavefunction
    :return: wavefunction at time step j+1
    """
    _,HDj,vals,vecs=sortedRetAll[j]

    # yj=psij-1/2*1j*dt*HDj@psij

    psijNext=np.zeros(M,dtype=complex)
    for m in range(0,M):
        Em=vals[m]
        vecm=vecs[:,m]
        psijNext+=(1-1/2*1j*dt*Em)/(1+1/2*1j*dt*Em)*np.vdot(vecm,psij)*vecm
    return psijNext

tEvoStart=datetime.now()
psi0=PsiValsAll[:,0]
psiNumericalAll=[psi0]
for j in range(0,N):
    psiCurr=psiNumericalAll[j]
    psiNext=oneStepEvolution(j,psiCurr)
    psiNumericalAll.append(psiNext)

tEvoEnd=datetime.now()

print("evolution time: ",tEvoEnd-tEvoStart)
# solutionNorm=[np.linalg.norm(psiNumericalAll[n]) for n in range(0,len(psiNumericalAll))]
# eps=1e-6
# for n in range(0,len(solutionNorm)):
#     if np.abs(solutionNorm[n]-1)>eps:
#         print(n)
diff=[]
for n in range(0,N+1):
    diff.append(np.linalg.norm(psiNumericalAll[n]-PsiValsAll[:,n],ord=2)*np.sqrt(dx))

plt.figure()
plt.plot(range(0,N+1),diff)
plt.savefig("diff.png")

np.savetxt("diff.txt",diff)