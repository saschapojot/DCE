import numpy as np
import re
import pandas as pd

#this script generates various dceNewxxx.py files to scan rows of inParamsNewyyy.csv
#in this computation, the evolution is from 0 to 5

dceFileName="dceNew"
suffix=".py"


group=5


inParamFileName="inParamsNew"+str(group)+".csv"
dfstr=pd.read_csv(inParamFileName)


fileIn=open(dceFileName+suffix,"r")
contents=fileIn.readlines()
lineSelectRow=0
lineGroupNum=0
for l in range(0,len(contents)):
    line=contents[l]
    if re.findall(r"^rowNum=\d+",line):
        lineSelectRow=l
    if re.findall(r"^group=\d+",line):
        lineGroupNum=l

nRow,nCol=dfstr.shape
for i in range(0,nRow):
    contents[lineSelectRow]="rowNum="+str(i)+"\n"
    contents[lineGroupNum]="group="+str(group)+"\n"
    outFileName="dceNew"+str(i)+".py"
    fileOut = open(outFileName, "w+")
    for oneline in contents:
        fileOut.write(oneline)
    fileOut.close()

for i in range(0,nRow):
    bashContents = []
    bashContents.append("#!/bin/bash\n")
    bashContents.append("#SBATCH -n 32\n")
    bashContents.append("#SBATCH -N 1\n")

    bashContents.append("#SBATCH -t 0-10:00\n")
    bashContents.append("#SBATCH -p CLUSTER\n")
    bashContents.append("#SBATCH --mem=80GB\n")

    bashContents.append("#SBATCH -o outdceNew" + str(i) + ".out\n")
    bashContents.append("#SBATCH -e outdceNew" + str(i) + ".err\n")
    bashContents.append("cd /home/cywanag/liuxi/Documents/pyCode/DCE\n")
    command="python3 dceNew"+str(i)+".py\n"
    bashContents.append(command)
    bsFileName = "./dceBash/dceNew" + str(i) + ".sh"
    fbsTmp = open(bsFileName, "w+")
    for oneline in bashContents:
        fbsTmp.write(oneline)
    fbsTmp.close()

