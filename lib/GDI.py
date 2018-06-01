"""
This will create a scaled GDI for the patients in the frame,
and then graph the average GDI for each patient on a graph, 
correlated with some other metric, such as the total amount of time
spent in the hospital for that patient, as well as the 
last recorded age for that patient. 

@Author: Anirrudh Krishnan
"""

import sys, os
from massager import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



massage = massager()
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))


#import data
def loadFullData(): 
    # csv_file = input("What is the name of the file that you would like to load?: ")
    csv_file = "CP_HAM"
    csv_file = "/data/" + csv_file + ".csv"
    csv_file = path + csv_file
    full_file = massage.loadCSV(path, csv_file)
    return full_file

def deleteNotAfflicted(df):
    df['Procedures.Side'] = df['Procedures.Side'].replace({'L':0})
    df['Procedures.Side'] = df['Procedures.Side'].replace({'R':1})
    df['MotionParams.Side'] = df['MotionParams.Side'].replace({'Right':1})
    df['MotionParams.Side'] = df['MotionParams.Side'].replace({'Left':0})

    """
    We will now delete sides that the procedures that were
    not done on. 
    """
    df = df[df['Procedures.Side'] == df['MotionParams.Side']]
        

    return df

def calculateResGDI(df):
    print('placeholder')

def invokeNormalization(df):
    """
    Using the technique of the Studentized residual, I will now
    normalize the GDI in realition with each other. 
    """
    print("Invoking Normalization Script. Normalizing GDI.")

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

patientGDI = {}
vector_gdi = []

df = loadFullData()
original_df_len = len(df)
df = df.dropna()
df = deleteNotAfflicted(df)

df['DeIdentifyNumber'] = df['DeIdentifyNumber'].astype(int)
patients = df.DeIdentifyNumber.unique()
print(patients)
# print(df['Procedures.Side'], df['MotionParams.Side'])
"""
Writing this for only places where
we know that the patient had a pre-op. 
"""
df = df[df['StudyType'] == "Pre-op"]

for p in patients:
    pFrame = df.loc[df['DeIdentifyNumber'] == p]
    # pFrame = pFrame[pFrame['StudyType_Pre-op'] != 0]
    if(pFrame.shape[0] > 0):
        
        gdiList = pFrame['GDI'].tolist()
        avgGdi = sum(gdiList)/len(gdiList)
        patientGDI[p] = avgGdi
        vector_gdi.append(avgGdi)

#Calculate Total Standard Deviation and mean for each person
gdi_mean = np.mean(vector_gdi)
gdi_stddev = np.std(vector_gdi)
print(len(df))

# print(gdi_mean)
# print(gdi_stddev)
# print(df)
running_mean = runningMeanFast(vector_gdi, original_df_len)
print(running_mean)
plt.plot(running_mean, x)
plt.show()