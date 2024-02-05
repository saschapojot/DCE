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


#this script plots photon and phonon number for many independent files (no segments)

class solution:
    def __init__(self):
        self.psiAll=np.zeros((1,1),dtype=complex)
        self.tStart=0
        self.tStop=0
        self.dt=0
        self.part=0
        self.rowNum=0
        self.group=0


part=0#all of them are 0th segments
group=2
N1=500
N2=500
L1=5
L2=10

dx1=2*L1/N1
dx2=2*L2/N2
x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
dtEst=0.002
tTot=5
M=int(tTot/dtEst)
dt=tTot/M
evoStart=0
eVoEnd=evoStart+tTot
##part of the operators
#construct H6

leftMat=sparse.diags(-2*np.ones(N1),offsets=0,format="lil",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=1,format="lil",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=-1,format="lil",dtype=complex)

H6=-1/(2*dx1**2)*sparse.kron(leftMat,sparse.eye(N2,dtype=complex,format="lil"),format="lil")

#compute <Nc>
tmp0=sparse.diags(x1ValsAll**2,format="lil")
IN2=sparse.eye(N2,dtype=complex,format="lil")
NcMat1=sparse.kron(tmp0,IN2)

# compute Nm
S2=sparse.diags(np.power(x2ValsAll,2),format="csc")
Q2=sparse.diags(-2*np.ones(N2),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=-1,format="csc",dtype=complex)

IN1=sparse.eye(N1,dtype=complex,format="lil")
NmPart1=sparse.kron(IN1,S2)
NmPart2=sparse.kron(IN1,Q2)
##read params csv
inParamFileName="inParamsNew"+str(group)+".csv"
dfstr=pd.read_csv(inParamFileName)
####load filenames
pklFileNames = []
rowVals = []
for file in glob.glob("./groupNew" + str(group) + "/*.pkl"):
    pklFileNames.append(file)
    matchRow = re.search(r"row(-?\d+(\.\d+)?)start", file)
    if matchRow:
        rowVals.append(matchRow.group(1))

def str2float(valList):
    ret = [int(strTmp) for strTmp in valList]
    return ret

rowVals = str2float(rowVals)

row_inds=np.argsort(rowVals)

pklFileNames=[pklFileNames[ind] for ind in row_inds]
rowVals=[row_inds[ind] for ind in row_inds]


outDir="./groupNew"+str(group)+"/"
path0="./groupNew"+str(group)+"/num/both/"
path1="./groupNew"+str(group)+"/num/photon/"
path2="./groupNew"+str(group)+"/wv/"
Path(path0).mkdir(parents=True, exist_ok=True)
Path(path1).mkdir(parents=True, exist_ok=True)
Path(path2).mkdir(parents=True, exist_ok=True)

for rowNum in range(0,len(rowVals)):
    print("file "+str(rowNum)+"============================")
    #load parameters in a row
    oneRow = dfstr.iloc[rowNum, :]
    j1H = int(oneRow.loc["j1H"])
    j2H = int(oneRow.loc["j2H"])
    g0 = oneRow.loc["g0"]
    omegam = oneRow.loc["omegam"]
    omegap = oneRow.loc["omegap"]
    er = oneRow.loc["er"]
    omegac = g0 * er
    thetaCoef = oneRow.loc["thetaCoef"]
    inDirPrefix = "./groupNew" + str(group) + "/row" + str(rowNum) + "start" + str(evoStart) + "stop" + str(
        eVoEnd) + "psiAllpart" + str(part)
    inPklFileName = inDirPrefix + ".pkl"
    tLoadStart = datetime.now()
    with open(inPklFileName,"rb") as fptr:
        wavefunctions=pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("loading time: ", tLoadEnd - tLoadStart)


    def avgNc(j):
        """

        :param j: time step j, wavefunction is Psi
        :return: number of photons for Psi
        """
        Psi = wavefunctions.psiAll[j, :]
        val = 1 / 2 * omegac * np.vdot(Psi, NcMat1 @ Psi) - 1 / 2 * np.vdot(Psi, Psi) + 1 / omegac * np.vdot(Psi,
                                                                                                             H6 @ Psi)

        return [j, val]


    def avgNm(j):
        """

        :param j:  time step j, wavefunction is Psi
        :return: number of phonons for Psi
        """
        Psi = wavefunctions.psiAll[j, :]

        val = 1 / 2 * omegam * np.vdot(Psi, NmPart1 @ Psi) - 1 / 2 * np.vdot(Psi, Psi) - 1 / (
                    2 * omegam * dx2 ** 2) * np.vdot(Psi, NmPart2 @ Psi)

        return [j, val]


    timeStepsAll = np.array([j for j in range(0, M + 1)])
    procNum = 48
    tNcStart = datetime.now()

    pool1 = Pool(procNum)
    retNc = pool1.map(avgNc, timeStepsAll)
    retNcSorted = sorted(retNc, key=lambda item: item[0])
    tNcEnd = datetime.now()

    print("Nc time: ", tNcEnd - tNcStart)
    tNmStart = datetime.now()
    pool2 = Pool(procNum)
    retNm = pool2.map(avgNm, timeStepsAll)
    retNmSorted = sorted(retNm, key=lambda item: item[0])
    tNmEnd = datetime.now()
    print("Nm time: ", tNmEnd - tNmStart)
    # plot

    NcVals = [np.abs(item[1]) for item in retNcSorted]
    NmVals = [np.abs(item[1]) for item in retNmSorted]

    plt.figure()

    plt.plot(timeStepsAll * dt, NcVals, color="blue", label="photon")
    plt.plot(timeStepsAll * dt, NmVals, color="red", label="phonon")
    xTicks = [0, 1 / 4 * tTot, 2 / 4 * tTot, 3 / 4 * tTot, tTot]
    xTicks = [round(val, 2) for val in xTicks]
    plt.xticks(xTicks)
    plt.title("$g_{0}=$" + str(g0) + ", initial phonon number = " + str(j2H) + ", $e^{r}=$" + str(er))
    plt.xlabel("time")
    plt.ylabel("number")
    plt.legend(loc="upper left")
    plt.savefig(path0 + "row" + str(rowNum) + "j1H" + str(j1H) + "j2H" + str(j2H) \
                + "g0" + str(g0) + "omegac" + str(omegac) + "omegam" + str(omegam) + "omegap" + str(
        omegap) + "er" + str(er) \
                + "thetaCoef" + str(thetaCoef) + "number.png")
    plt.close()

    plt.figure()

    plt.plot(timeStepsAll * dt, NcVals, color="blue", label="photon")
    xTicks = [0, 1 / 4 * tTot, 2 / 4 * tTot, 3 / 4 * tTot, tTot]
    xTicks = [round(val, 2) for val in xTicks]
    plt.xticks(xTicks)
    plt.title("$g_{0}=$" + str(g0) + ", initial phonon number = " + str(j2H) + ", $e^{r}=$" + str(er))
    plt.xlabel("time")
    plt.ylabel("photon number")
    plt.legend(loc="upper left")

    plt.savefig(path1 + "row" + str(rowNum) + "j1H" + str(j1H) + "j2H" + str(j2H) \
                + "g0" + str(g0) + "omegac" + str(omegac) + "omegam" + str(omegam) + "omegap" + str(
        omegap) + "er" + str(er) \
                + "thetaCoef" + str(thetaCoef) + "photon.png")
    plt.close()


    def psi2Mat(psi):
        """

        :param psi: wavefunction at one time step
        :return: 2d representation of psi
        """
        mat = np.zeros((N1, N2), dtype=complex)
        for n1 in range(0, N1):
            for n2 in range(0, N2):
                mat[n1, n2] = psi[n1 * N2 + n2]
        return mat


    j2Plot = [0, -1]
    for j in j2Plot:
        mat = np.abs(psi2Mat(wavefunctions.psiAll[j, :]))
        plt.figure()
        plt.imshow(mat)
        plt.title("$t=$" + str((j % (M + 1) * dt)))
        plt.colorbar()
        plt.savefig(path2 + "row" + str(rowNum) + "j1H" + str(j1H) + "j2H" + str(j2H) \
                    + "g0" + str(g0) + "omegac" + str(omegac) + "omegam" + str(omegam) + "omegap" + str(
            omegap) + "er" + str(er) \
                    + "thetaCoef" + str(thetaCoef) + "tStep" + str(j) + ".png")
        plt.close()




