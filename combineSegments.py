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
#this script loads different computation segments and compute physical observables
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
group=5
inParamFileName="inParams"+str(group)+".csv"

dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]


j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])
g0=oneRow.loc["g0"]

omegac=oneRow.loc["omegac"]
omegam=oneRow.loc["omegam"]
omegap=oneRow.loc["omegap"]
er=oneRow.loc["er"]
lmd=(er**2-1/er**2)/(er**2+1/er**2)*(omegam-omegap)
thetaCoef=oneRow.loc["thetaCoef"]
theta=np.pi*thetaCoef
Deltam=omegam-omegap
N1=500
N2=500
L1=5
L2=10

dx1=2*L1/N1
dx2=2*L2/N2
x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])



psiAll=np.zeros((2,2),dtype=complex)

#construct number operators
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
    Psi=psiAll[j,:]
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
    Psi = psiAll[j,:]

    val=1/2*omegam*np.vdot(Psi,NmPart1@Psi)-1/2*np.vdot(Psi,Psi)-1/(2*omegam*dx2**2)*np.vdot(Psi,NmPart2@Psi)

    return [j,val]



####load data
pklFileNames = []
startVals = []
for file in glob.glob("./group" + str(group) + "/*.pkl"):
    pklFileNames.append(file)
    matchStart=re.search(r"start(-?\d+(\.\d+)?)stop",file)
    if matchStart:
        startVals.append(matchStart.group(1))

val0 = (len(pklFileNames) - len(startVals)) ** 2
if val0 != 0:
    raise ValueError("unequal length.")


def str2float(valList):
    ret = [float(strTmp) for strTmp in valList]
    return ret


startVals = str2float(startVals)

start_inds = np.argsort(startVals)

pklFileNames = [pklFileNames[ind] for ind in start_inds]
NcList=[]
NmList=[]
procNum=48
psiFirstLast=[]#initial and final value of wavefunction
for i in range(0,len(pklFileNames)):
    print("load file "+str(i))
    onePklFile=pklFileNames[i]
    tLoadStart=datetime.now()
    with open(onePklFile,"rb") as fptr:
        wvTmp=pickle.load(fptr)
    tLoadEnd=datetime.now()
    print("loading time: ",tLoadEnd-tLoadStart)
    if i==0:
        psiFirstLast.append(wvTmp.psiAll[0,:])
    if i==len(pklFileNames)-1:
        psiFirstLast.append(wvTmp.psiAll[-1:])
    M=len(wvTmp.psiAll)-1
    dt=wvTmp.dt
    timeStepsAll = np.array([j for j in range(0, M + 1)])
    psiAll=wvTmp.psiAll
    #compute Nc
    tNcStart = datetime.now()
    pool1=Pool(procNum)
    retNc=pool1.map(avgNc,timeStepsAll)
    retNcSorted = sorted(retNc, key=lambda item: item[0])
    NcTmp=[np.abs(item[1]) for item in retNcSorted]
    NcList.append(NcTmp)
    tNcEnd = datetime.now()
    print("Nc time: ",tNcEnd-tNcStart)
    #compute Nm
    tNmStart = datetime.now()
    pool2 = Pool(procNum)
    retNm = pool2.map(avgNm, timeStepsAll)
    retNmSorted = sorted(retNm, key=lambda item: item[0])
    NmTmp=[np.abs(item[1]) for item in retNmSorted]
    NmList.append(NmTmp)
    tNmEnd = datetime.now()
    print("Nm time: ",tNmEnd-tNmStart)

#combine values
# for item in NcList:
#     print("Nc segment length="+str(len(item)))
# for item in NmList:
#     print("Nm segment length="+str(len(item)))
NcCombined=[]

NcCombined+=NcList[0]
for i in range(1,len(NcList)):
    tmpList=NcList[i]
    NcCombined+=tmpList[1:]
# print("len(NcCombined)="+str(len(NcCombined)))

NmCombined=[]
NmCombined+=NmList[0]
for i in range(1,len(NmList)):
    tmpList=NmList[i]
    NmCombined+=tmpList[1:]
# print("len(NmCombined)="+str(len(NmCombined)))
#outdir
path0="./group"+str(group)+"/num/both/"
path1="./group"+str(group)+"/num/photon/"
path2="./group"+str(group)+"/wv/"
Path(path0).mkdir(parents=True, exist_ok=True)
Path(path1).mkdir(parents=True, exist_ok=True)
Path(path2).mkdir(parents=True, exist_ok=True)

print(NmCombined)
#plt phonon and photon
tValsAll=[dt*j for j in range(0,len(NcCombined))]
# print("len(NmCombined+)="+str(len(tValsAll)))
plt.plot(tValsAll,NcCombined,color="blue",label="photon")
plt.plot(tValsAll,NmCombined,color="red",label="phonon")
tTot=max(tValsAll)
xTicks=[0,1/4*tTot,2/4*tTot,3/4*tTot,tTot]
xTicks=[round(val,2) for val in xTicks]
plt.xticks(xTicks)
plt.title("$g_{0}=$"+str(g0)+", initial phonon number = "+str(j2H))
plt.xlabel("time")
plt.ylabel("number")
plt.legend(loc="upper left")
plt.savefig(path0+"row"+str(rowNum)+"j1H"+str(j1H)+"j2H"+str(j2H)\
    +"g0"+str(g0)+"omegac"+str(omegac)+"omegam"+str(omegam)+"omegap"+str(omegap)+"er"+str(er)\
    +"thetaCoef"+str(thetaCoef)+"number.png")

plt.close()

#plt photon
plt.figure()
plt.plot(tValsAll,NcCombined,color="blue",label="photon")
plt.xticks(xTicks)
plt.title("$g_{0}=$"+str(g0)+", initial phonon number = "+str(j2H))
plt.xlabel("time")
plt.ylabel("photon number")
plt.legend(loc="upper left")
plt.savefig(path1+"row"+str(rowNum)+"j1H"+str(j1H)+"j2H"+str(j2H)\
    +"g0"+str(g0)+"omegac"+str(omegac)+"omegam"+str(omegam)+"omegap"+str(omegap)+"er"+str(er)\
    +"thetaCoef"+str(thetaCoef)+"photon.png")

def psi2Mat(psi):
    """

    :param psi: wavefunction at one time step
    :return: 2d representation of psi
    """
    mat=np.zeros((N1,N2),dtype=complex)
    for n1 in range(0,N1):
        for n2 in range(0,N2):
            mat[n1,n2]=psi[n1*N2+n2]
    return mat

indPlot=[0,-1]
for j in indPlot:
    mat=np.abs(psiFirstLast[j])
    plt.figure()
    plt.imshow(mat)
    plt.title("$t=$" + str(tValsAll[j]))
    plt.colorbar()
    plt.savefig(path2+"row"+str(rowNum)+"j1H"+str(j1H)+"j2H"+str(j2H)\
    +"g0"+str(g0)+"omegac"+str(omegac)+"omegam"+str(omegam)+"omegap"+str(omegap)+"er"+str(er)\
    +"thetaCoef"+str(thetaCoef)+ "tStep" + str(j) + ".png")
    plt.close()