import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from multiprocessing import Pool
import math
from scipy.sparse.linalg import spsolve
# import scipy.linalg
from scipy.special import hermite
from datetime import datetime
import copy
from pathlib import Path
from scipy import sparse
# from scipy.linalg import ishermitian
# from scipy.sparse.linalg import inv
import pickle
from mpmath import fac
#this script uses inParamsNewxxx.csv

rowNum=0


group=2


inParamFileName="inParamsNew"+str(group)+".csv"

dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]


j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

def H(n,x):
    """

    :param n: order of Hermite polynomial
    :param x:
    :return: value of polynomial at x
    """
    return hermite(n)(x)/np.sqrt(fac(n))


g0=oneRow.loc["g0"]


omegam=oneRow.loc["omegam"]
omegap=oneRow.loc["omegap"]
er=oneRow.loc["er"]#magnification
omegac=g0*er
# print("er="+str(er))
lmd=(er**2-1/er**2)/(er**2+1/er**2)*(omegam-omegap)
# print("lambda="+str(lmd))
# print("Deltam="+str(omegam-omegap))
thetaCoef=oneRow.loc["thetaCoef"]
theta=np.pi*thetaCoef
Deltam=omegam-omegap


tStart=0
tStop=1
tTot=(tStop-tStart)

N1=500

N2=2048
L1=5
L2=40


dx1=2*L1/N1
dx2=2*L2/N2
# print(dx2)
dtEst=0.002
M=int(tTot/dtEst)
dt=tTot/M

x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])

#construct H0
IN1N2=sparse.eye(N1*N2,dtype=complex,format="csc")
H0=(-1/2*omegac-1/2*Deltam)*IN1N2

D2=sparse.diags(x2ValsAll,format="csc")
S2=sparse.diags(np.power(x2ValsAll,2),format="csc")
#diagD2=diag(D2,D2,...,D2)
blockD2=[D2 for n1 in range(0,N1)]
diagD2=sparse.block_diag(blockD2,format="csc",dtype=complex)
#diagS2=diag(S2,S2,...,S2)
blockS2=[S2 for n1 in range(0, N1)]
diagS2=sparse.block_diag(blockS2,format="csc",dtype=complex)


def H1Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H1 at time step j
    """
    tj = j * dt
    return -1 / 2 * g0 * np.sqrt(2 * omegam) * np.cos(omegap * (tj + 1 / 2 * dt)) * diagD2


#construct H2
IN2=sparse.eye(N2,dtype=complex,format="csc")
blockH2=[IN2*x1ValsAll[n1]**2 for n1 in range(0,N1)]
diagH2=sparse.block_diag(blockH2,format="csc",dtype=complex)
H2=1/2*omegac**2*diagH2

#construct H3
H3=(1/2*lmd*omegam*np.cos(theta)+1/2*Deltam*omegam)*diagS2


blockH4=[D2*x1ValsAll[n1]**2 for n1 in range(0,N1)]
diagH4=sparse.block_diag(blockH4,format="csc",dtype=complex)

def H4Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H4 at time step j
    """

    tj=j*dt
    return g0 * omegac * np.sqrt(2 * omegam) * np.cos(omegap * (tj + 1 / 2 * dt)) * diagH4



P2Ones=np.ones(N2-1)
P2=sparse.diags(P2Ones,offsets=1,format="csc",dtype=complex)\
    +sparse.diags(-P2Ones, offsets=-1,format="csc",dtype=complex)

blockH5=[P2 for n1 in range(0,N1)]
diagH5=sparse.block_diag(blockH5,format="csc",dtype=complex)


def H5Mat(j):
    """

    :param j: time  step j
    :return: value of matrix H5 at time step j
    """
    tj=j*dt

    A5j=-1j*g0*omegac*np.sqrt(2/omegam)*np.sin(omegap*(tj+1/2*dt))*diagH2\
        +1j*1/2*g0*np.sqrt(2/omegam)*np.sin(omegap*(tj+1/2*dt))*IN1N2

    return A5j*1/(2*dx2)*diagH5



#construct H6

leftMat=sparse.diags(-2*np.ones(N1),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=-1,format="csc",dtype=complex)




H6=-1/(2*dx1**2)*sparse.kron(leftMat,sparse.eye(N2,dtype=complex,format="csc"),format="csc")


#construct H7

Q2=sparse.diags(-2*np.ones(N2),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=-1,format="csc",dtype=complex)

blockH7=[Q2 for n1 in range(0,N1)]

diagH7=sparse.block_diag(blockH7,format="csc",dtype=complex)
H7=(-Deltam/(2*omegam)+lmd*np.cos(theta)/(2*omegam))*1/(dx2**2)*diagH7

#construct H8
antiCommD2P2=D2@P2+P2@D2
H8=1j*lmd*np.sin(theta)/(4*dx2)*sparse.kron(sparse.eye(N1,dtype=complex,format="csc"),antiCommD2P2)


def psi0(n1n2):
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
ret0=pool0.map(psi0,n1n2All)
Psi0=np.zeros(N1*N2,dtype=complex)
for item in ret0:
    n1,n2,val=item
    Psi0[n1*N2+n2]=val
Psi0/=np.linalg.norm(Psi0,ord=2)
tInitEnd=datetime.now()

print("Initialization of Psi: ",tInitEnd-tInitStart)



# def evoMat(j):
#     """
#
#     :param j: time step j
#     :return: (1-1/2 * i*dt *H)/(1+1/2*i*dt*H)
#     """
#     HDRj = H0 + H1Mat(j) + H2 + H3 + H4Mat(j) + H5Mat(j) + H6 + H7 + H8
#
#     mat=(IN1N2-1/2*1j*dt*HDRj)@inv(IN1N2+1/2*1j*dt*HDRj)
#
#     return [j,mat]
def HMat(j):
    """

    :param j: time step j
    :return: HDRj
    """
    HDRj = H0 + H1Mat(j) + H2 + H3 + H4Mat(j) + H5Mat(j) + H6 + H7 + H8
    return [j,HDRj]


tStepsAll=[j for j in range(0,M)]
tHStart=datetime.now()
pool0=Pool(procNum)
retHAll=pool0.map(HMat,tStepsAll)
retHAllSorted=sorted(retHAll,key=lambda item:item[0])
tHEnd=datetime.now()

print("Construct All H time: ",tHEnd-tHStart)

tEvolutionStart=datetime.now()


class solution:
    def __init__(self):
        self.psiAll=np.zeros((M+1,N1*N2),dtype=complex)
        self.tStart=tStart
        self.tStop=tStop
        self.dt=dt
        self.part=0
        self.rowNum=rowNum
        self.group=group
wavefunctions=solution()
wavefunctions.psiAll[0,:]=Psi0
# PsiAll=[copy.deepcopy(Psi0)]
for j in range(0,M):

    if j%10==0:
        t500Multiple=datetime.now()
        print("step "+str(j)," time: ",t500Multiple-tEvolutionStart)

    PsiCurr=wavefunctions.psiAll[j,:]
    Htmp=retHAllSorted[j][1]
    y0=spsolve(IN1N2+1/2*1j*dt*Htmp,PsiCurr)
    PsiNext=(IN1N2-1/2*1j*dt*Htmp)@y0
    # PsiAll.append(PsiNext)
    wavefunctions.psiAll[j+1,:]=PsiNext


# outData=np.array(PsiAll).T
# dtFrm=pd.DataFrame(data=outData)
# outDirPrefix="./omegac"+str(omegac)+"omegam"+str(omegam)+"omegap"+str(omegap)+"er"+str(er)+"theta"+str(theta/np.pi)+"pi"+"/"
outDirPrefix= "groupNew"+str(group)+"/"
Path(outDirPrefix).mkdir(parents=True, exist_ok=True)

# dtFrm.to_csv(outDirPrefix+"PsiAll.csv",index=False,header=False)

tEvolutionEnd=datetime.now()

print("evolution time: ",tEvolutionEnd-tEvolutionStart)

outPklFileName=outDirPrefix+"row"+str(rowNum)+"start"+str(tStart)+"stop"+str(tStop)+"psiAllpart"+str(wavefunctions.part)\
               +"N1"+str(N1)+"N2"+str(N2)+"L1"+str(L1)+"L2"+str(L2)+".pkl"
with open(outPklFileName,"wb") as fptr:
    pickle.dump(wavefunctions,fptr,pickle.HIGHEST_PROTOCOL)

##############memory usage
import psutil
process = psutil.Process()
print("mem total="+str((process.memory_info().rss)/1024**3)+"GB")