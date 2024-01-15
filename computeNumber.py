import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import  datetime
from scipy import sparse
import matplotlib.pyplot as plt

inFileName="./out1/"


dfstr=pd.read_csv(inFileName+"PsiAll.csv",header=None)
print("finished reading")
nRow,nCol=dfstr.shape
#nCol is total time step number
# print(nRow)
# print(nCol)
# print(complex(dfstr.iloc[1,2]))
PsiAll=np.zeros((nRow,nCol),dtype=complex)
def str2complex(ij):
    """

    :param ij: [i,j]
    :return: convert [i,j]-th element of dfstr from str to complex
    """
    i,j=ij
    # try:
    #     ret=[i,j,complex((dfstr.iloc[i,j]).replace(" ", ""))]
    #     return ret
    # except ValueError as e:
    #     print([i,j])


    return [i,j,complex((dfstr.iloc[i,j]).replace(" ", ""))]

def str2complexSerial(i,j):
    """

    :param i: row
    :param j: col
    :return: element [i,j]
    """
    return complex((dfstr.iloc[i,j]).replace(" ", ""))

# ijAll=[[i,j] for i in range(0,nRow) for j in range(0,nCol)]

procNum=48

# pool0=Pool(procNum)

t2ComplexStart=datetime.now()
for i in range(0,nRow):
    for j in range(0,nCol):
        PsiAll[i,j]=str2complexSerial(i,j)

# ret0=pool0.map(str2complex,ijAll)



# for item in ret0:
#     i,j,val=item
#     PsiAll[i,j]=val

t2ComplexEnd=datetime.now()

print("str to complex time: ",t2ComplexEnd-t2ComplexStart)
omegac=100
omegam=3
N1=500
N2=500
L1=5
L2=5
dx1=2*L1/N1
dx2=2*L2/N2
dtEst=0.002
tTot=1
M=int(tTot/dtEst)
dt=tTot/M

x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
#construct H6

leftMat=sparse.diags(-2*np.ones(N1),offsets=0,format="lil",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=1,format="lil",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=-1,format="lil",dtype=complex)

H6=-1/(2*dx1**2)*sparse.kron(leftMat,sparse.eye(N2,dtype=complex,format="lil"),format="lil")

#compute <Nc>
tmp0=sparse.diags(x1ValsAll**2,format="lil")
IN2=sparse.eye(N2,dtype=complex,format="lil")
NcMat1=sparse.kron(tmp0,IN2)

def avgNc(j):
    """

    :param j: time step j, wavefunction is Psi
    :return: number of photons for Psi
    """
    Psi=PsiAll[:,j]
    val=1/2*omegac*np.vdot(Psi,NcMat1@Psi)-1/2*np.vdot(Psi,Psi)+1/omegac*np.vdot(Psi,H6@Psi)

    return [j,val]

# compute Nm
S2=sparse.diags(np.power(x2ValsAll,2),format="csc")
Q2=sparse.diags(-2*np.ones(N2),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=-1,format="csc",dtype=complex)

IN1=sparse.eye(N1,dtype=complex,format="lil")
NmPart1=sparse.kron(IN1,S2)
NmPart2=sparse.kron(IN1,Q2)

def avgNm(j):
    """

    :param j:  time step j, wavefunction is Psi
    :return: number of phonons for Psi
    """
    Psi = PsiAll[:, j]

    val=1/2*omegam*np.vdot(Psi,NmPart1@Psi)-1/2*np.vdot(Psi,Psi)-1/(2*omegam*dx2**2)*np.vdot(Psi,NmPart2@Psi)

    return [j,val]


timeStepsAll=[j for j in range(0,nCol)]

tNcStart=datetime.now()

pool1=Pool(procNum)
retNc=pool1.map(avgNc,timeStepsAll)
retNcSorted=sorted(retNc,key=lambda item: item[0])
tNcEnd=datetime.now()

print("Nc time: ",tNcEnd-tNcStart)

tNmStart=datetime.now()
pool2=Pool(procNum)
retNm=pool2.map(avgNm,timeStepsAll)
retNmSorted=sorted(retNm,key=lambda item: item[0])
tNmEnd=datetime.now()
print("Nm time: ",tNmEnd-tNmStart)

#plot

NcVals=[np.abs(item[1]) for item in retNcSorted]
NmVals=[np.abs(item[1]) for item in retNmSorted]

plt.figure()

plt.plot(timeStepsAll,NcVals,color="blue",label="photon")
plt.plot(timeStepsAll,NmVals,color="red",label="phonon")
plt.legend(loc="upper left")
plt.savefig(inFileName+"n.png")