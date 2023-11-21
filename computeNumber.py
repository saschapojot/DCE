import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import  datetime

inFileName="./omegac100omegam3omegap2.95er50theta0.14285714285714285pi/"


dfstr=pd.read_csv(inFileName+"PsiAll.csv",header=None)

nRow,nCol=dfstr.shape

PsiAll=np.zeros((nRow,nCol),dtype=complex)
def str2complex(ij):
    """

    :param ij: [i,j]
    :return: convert [i,j]-th element of dfstr from str to complex
    """
    i,j=ij
    return [i,j,complex(dfstr.iloc[i,j])]

ijAll=[[i,j] for i in range(0,nRow) for j in range(0,nCol)]

procNum=48

pool0=Pool(procNum)

t2ComplexStart=datetime.now()

ret0=pool0.map(str2complex,ijAll)



for item in ret0:
    i,j,val=item
    PsiAll[i,j]=val

t2ComplexEnd=datetime.now()

print("str to complex time: ",t2ComplexEnd-t2ComplexStart)

