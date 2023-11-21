import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import math
from scipy.special import hermite
from datetime import datetime
from scipy.linalg import block_diag
import copy
from pathlib import Path
# from scipy.linalg import ishermitian

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
N1=500
N2=500
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
diagD2=block_diag(*allD2)#diagD2=diag(D2,D2,...,D2)
diagS2=block_diag(*allS2)#diagS2=diag(S2,S2,...,S2)

def H1Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H1 at time step j
    """
    tj=j*dt
    return -1/2*g0*np.sqrt(2*omegam)*np.cos(omegap*(tj+1/2*dt))*diagD2

#construct H2
IN2=np.eye(N2,dtype=complex)
blockH2=[IN2*x1ValsAll[n1]**2 for n1 in range(0,N1)]
diagH2=block_diag(*blockH2)
H2=1/2*omegac**2*diagH2


#construct H3
H3=(1/2*lmd*omegam*np.cos(theta)+1/2*Deltam*omegam)*diagS2


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



#construct H6

leftMat=np.diag(-2*np.ones(N1),k=0)+np.diag(np.ones(N1-1),k=1)+np.diag(np.ones(N1-1),k=-1)

H6=-1/(2*dx1**2)*np.kron(leftMat,np.eye(N2,dtype=complex))


#construct H7

Q2=np.diag(-2*np.ones(N2),k=0)+np.diag(np.ones(N2-1),k=1)+np.diag(np.ones(N2-1),k=-1)
blockH7=[Q2 for n1 in range(0,N1)]
diagH7=block_diag(*blockH7)
H7=(-Deltam/(2*omegam)+lmd*np.cos(theta)/(2*omegam))*1/(dx2**2)*diagH7

#construct H8
antiCommD2P2=D2@P2+P2@D2
H8=1j*lmd*np.sin(theta)/(4*dx2)*np.kron(np.eye(N1,dtype=complex),antiCommD2P2)


def psi(n1n2):
    """

    :param n1n2: [n1,n2], n1, n2 are positions in x1, x2 array
    :return: initial value of wavefunction with position: [n1,n2, psi]
    """
    n1,n2=n1n2
    x1=x1ValsAll[n1]
    x2=x2ValsAll[n2]
    funcVal= np.exp(-1 / 2 * omegac * x1 ** 2) * H(j1H, np.sqrt(omegac)*x1)\
             * np.exp(-1 / 2 * omegam * x2 ** 2) * H(j2H,np.sqrt(omegam)*x2)

    return [n1,n2,funcVal]


n1n2All=[[n1,n2] for n1 in range(0,N1) for n2 in range(0,N2)]

procNum=48

pool0=Pool(procNum)
tInitStart=datetime.now()
ret0=pool0.map(psi,n1n2All)



Psi0=np.zeros(N1*N2,dtype=complex)
for item in ret0:
    n1,n2,val=item
    Psi0[n1*N2+n2]=val
Psi0/=np.linalg.norm(Psi0,ord=2)
tInitEnd=datetime.now()

print("Initialization of Psi: ",tInitEnd-tInitStart)


def eigj(j):
    """

    :param j: time step j
    :return: eigenvalues and eigenvectors of HDRj
    """

    HDRj=H0+H1Mat(j)+H2+H3+H4Mat(j)+H5Mat(j)+H6+H7+H8

    vals,vecs=np.linalg.eigh(HDRj)
    return [j,vals,vecs]




tStepsAll=[j for j in range(0,M)]
tEigStart=datetime.now()

pool1=Pool(procNum)

ret1=pool1.map(eigj,tStepsAll)

tEigEnd=datetime.now()

print("eig time: ",tEigEnd-tEigStart)

sortedRet1=sorted(ret1,key=lambda item: item[0])


def oneStepEvolution(j,Psij):
    """

    :param j: current step
    :param Psij: current wavefunction
    :return: wavefunction at time step j+1
    """

    _,vals,vecs=sortedRet1[j]
    PsiNext=np.zeros(N1*N2,dtype=complex)
    for l in range(0,len(vals)):
        El=vals[l]
        vecl=vecs[:,l]
        PsiNext+=(1-1/2*1j*dt*El)/(1+1/2*1j*dt*El)*np.vdot(vecl,Psij)*vecl

    return PsiNext



tEvoStart=datetime.now()

PsiAll=[copy.deepcopy(Psi0)]
for j in range(0,M):
    PsiCurr=PsiAll[j]
    PsiNext=oneStepEvolution(j,PsiCurr)
    PsiAll.append(PsiNext)


tEvoEnd=datetime.now()

print("evolution time: ",tEvoEnd-tEvoStart)

outData=np.array(PsiAll).T
dtFrm=pd.DataFrame(data=outData)




outDirPrefix="./omegac"+str(omegac)+"omegam"+str(omegam)+"omegap"+str(omegap)+"er"+str(er)+"theta"+str(theta/np.pi)+"pi"+"/"
Path(outDirPrefix).mkdir(parents=True, exist_ok=True)

dtFrm.to_csv(outDirPrefix+"PsiAll.csv",index=False,header=False)













































































