import pickle
# from copy import deepcopy
import numpy as np
from datetime import datetime
from scipy import sparse
import matplotlib.pyplot as plt
import glob
import re
from multiprocessing import Pool
import pandas as pd
from pathlib import Path

#this script computes squeezed phonon  number, after loading different computation segments
class solution:
    def __init__(self):
        self.psiAll=np.zeros((1,1),dtype=complex)
        self.tStart=0
        self.tStop=0
        self.dt=0
        self.part=0
        self.rowNum=0
        self.group=0

rowNum=0
group=3
inParamFileName="inParamsNew"+str(group)+".csv"

dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]


j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])
g0=oneRow.loc["g0"]

# omegac=oneRow.loc["omegac"]
omegam=oneRow.loc["omegam"]
omegap=oneRow.loc["omegap"]

er=oneRow.loc["er"]
r=np.log(er)
omegac=g0*er
lmd=(er**2-1/er**2)/(er**2+1/er**2)*(omegam-omegap)
thetaCoef=oneRow.loc["thetaCoef"]
theta=np.pi*thetaCoef
Deltam=omegam-omegap
N1=500
N2=1024
L1=5
L2=20

dx1=2*L1/N1
dx2=2*L2/N2
x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
psiAll=np.zeros((2,2),dtype=complex)


#construct D2
D2=sparse.diags(x2ValsAll,format="csc")
#construct S2
S2=sparse.diags(np.power(x2ValsAll,2),format="csc")
#diagD2=diag(D2,D2,...,D2)
blockD2=[D2 for n1 in range(0,N1)]
diagD2=sparse.block_diag(blockD2,format="csc",dtype=complex)

#diagS2=diag(S2,S2,...,S2)
blockS2=[S2 for n1 in range(0, N1)]
diagS2=sparse.block_diag(blockS2,format="csc",dtype=complex)
ns2=(1/2*omegam*np.cos(theta)*np.sinh(2*r)+1/2*omegam*np.cosh(2*r))*diagS2
IN1N2=sparse.eye(N1*N2,dtype=complex,format="csc")
ns3=-1/2*IN1N2
P2Ones=np.ones(N2-1)
P2=sparse.diags(P2Ones,offsets=1,format="csc",dtype=complex)\
    +sparse.diags(-P2Ones, offsets=-1,format="csc",dtype=complex)

Q2=sparse.diags(-2*np.ones(N2),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=-1,format="csc",dtype=complex)
blockQ2=[Q2 for n1 in range(0,N1)]
diagQ2=sparse.block_diag(blockQ2,format="csc",dtype=complex)
ns0=(1/(2*omegam)*np.cos(theta)*np.sinh(2*r)-1/(2*omegam)*np.cosh(2*r))/dx2**2*diagQ2


antiCommD2P2=D2@P2+P2@D2
ns1=1j*np.sin(theta)*np.sinh(2*r)/(4*dx2)*sparse.kron(sparse.eye(N1,dtype=complex,format="csc"),antiCommD2P2)

ns=ns0+ns1+ns2+ns3

def avgns(j):
    """

    :param j: time step j, wavefunction is Psi
    :return: number of squeezed phonons for Psi
    """
    Psi = psiAll[j, :]

    val=np.abs(np.vdot(Psi,ns@Psi))
    return [j,val]


####load data
pklFileNames = []
startVals = []
for file in glob.glob("./groupNew" + str(group) + "/*.pkl"):
    pklFileNames.append(file)
    matchStart=re.search(r"start(-?\d+(\.\d+)?)stop",file)
    if matchStart:
        startVals.append(matchStart.group(1))


def str2float(valList):
    ret = [float(strTmp) for strTmp in valList]
    return ret


startVals = str2float(startVals)

start_inds = np.argsort(startVals)
pklFileNames = [pklFileNames[ind] for ind in start_inds]
nsList=[]
procNum=48
for i in range(0,len(pklFileNames)):
    print("load file " + str(i))
    onePklFile = pklFileNames[i]
    tLoadStart = datetime.now()
    with open(onePklFile, "rb") as fptr:
        wvTmp = pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("loading time: ", tLoadEnd - tLoadStart)
    M = len(wvTmp.psiAll) - 1
    dt = wvTmp.dt
    timeStepsAll = np.array([j for j in range(0, M + 1)])
    psiAll = wvTmp.psiAll
    tnsStart = datetime.now()
    pool1 = Pool(procNum)
    retns=pool1.map(avgns,timeStepsAll)
    retnsSorted = sorted(retns, key=lambda item: item[0])
    nsTmp=[item[1] for item in retnsSorted]
    nsList.append(nsTmp)
    tnsEnd=datetime.now()
    print("ns time: ",tnsEnd-tnsStart)



#combine values
nsCombined=[]
nsCombined+=nsList[0]
for i in range(1,len(nsList)):
    tmpList=nsList[i]
    nsCombined+=tmpList[1:]

#outdir
path0="./groupNew"+str(group)+"/num/squeezed/"
Path(path0).mkdir(parents=True, exist_ok=True)

#plot
tValsAll=[dt*j for j in range(0,len(nsCombined))]

plt.figure()
plt.plot(tValsAll,nsCombined,color="green",label="squeezed phonon")
tTot=max(tValsAll)
xTicks=[0,1/4*tTot,2/4*tTot,3/4*tTot,tTot]
xTicks=[round(val,2) for val in xTicks]
plt.title("$g_{0}=$"+str(g0)+", initial phonon number = "+str(j2H)+", $e^{r}=$"+str(er))
plt.xlabel("time")
plt.ylabel("number")
plt.legend(loc="upper left")
plt.savefig(path0+"row"+str(rowNum)+"j1H"+str(j1H)+"j2H"+str(j2H)\
    +"g0"+str(g0)+"omegac"+str(omegac)+"omegam"+str(omegam)+"omegap"+str(omegap)+"er"+str(er)\
    +"thetaCoef"+str(thetaCoef)+"squeezednumber.png")

plt.close()