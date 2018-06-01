"""
This script graphs patients who only have a post and preop.
Information is only graphed from the surgery date onward.
Author: Spencer Christensen
"""



import pandas as pd
import datetime
import numpy as np
import scipy
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from datetime import *
from lib.massager import *

def getfilepath():
    """
    Get the filepath of the csv file that needs to be loaded.
    :return:
    """
    filepath = input('Please Drag and Drop your CSV File here: ')
    df = pd.read_excel(filepath)
    return df


def main():
    #Get filepath, drop GDI's under 20 as cleansing measure, create new columns from categorical variables.
    df = getfilepath()
    df = df[df['GDI'] > 20]
    cat = list(df.iloc[:, [1,3,4,5,9]].copy())
    dmDF = pd.get_dummies(df, columns= cat)
    dmDF = dmDF.drop(['Comment'], axis=1)
    dmDF = dmDF.drop(['Trial_GcdFile'], axis=1)
    dmDF = dmDF.dropna()

    #TODO - Write drop function for people with multiple surgeries.
    drop = [94850, 46577, 58885, 70712, 55181, 16273, 98666]
    for d in drop:
       dmDF = dmDF[dmDF['DeIdentifyNumber'] != d]

    #Get unique patient ids
    patients = dmDF.DeIdentifyNumber.unique()

    """
    Find the surgery date for all patients.
    We assume only one date of surgery, but this structure can support multiple dates.
    """
    sx = {}
    for p in patients:
        x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
        x = x[x['StudyType_Pre-op'] == 1]
        sxDate = x['TestDate'].unique()
        sx[p] = sxDate

    #Get a list of all visit dates
    vis = []
    for p in patients:
        x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
        vis.append(x['TestDate'].unique())

    #Dataset has 3 rows for one test. Average all rows together by defining what columns require averaging.
    avCols = ['WalkingSpeedMetersPerSec', 'StepLengthM', 'FootOffPct', 'PelvicTiltMax', 'PelvicTiltMin', 'HipFlexionMin',
              'KneeFlexionMin', 'KneeFlexionPeak', 'KneeFlexionAtIC', 'AnkleDorsiPlantarPeakPF', 'GDI']

    #Create new dataframe to hold new averaged rows.
    newDF = pd.DataFrame(columns=list(dmDF))



    """
    The following loop appends the new averaged rows to newDF.
    It also cleanses any records where a patient doesn't have 3 tests in a visit,
    which is the expected format of the data.

    """
    for p in patients:
        x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
        visits = x['TestDate'].unique()

        for v in visits:
            visitDF_L = x[x['TestDate'] == v]
            leftDF = visitDF_L[visitDF_L['MotionParams.Side_Left'] == 1]

            if (leftDF.shape[0]%3 != 0 or leftDF.shape[0] == 0):
                break

            else:
                tempRow = leftDF.iloc[0]
                for cols in avCols:
                    avg = leftDF[cols].mean()
                    tempRow[cols] = avg
            newDF.loc[len(newDF)] = tempRow

        for v in visits:
            visitDF_R = x[x['TestDate'] == v]
            rightDF = visitDF_R[visitDF_R['MotionParams.Side_Right'] == 1]

            if (rightDF.shape[0]%3 != 0 or rightDF.shape[0] == 0):
                break

            else:
                tempRow = rightDF.iloc[0]

                for cols in avCols:
                    avg = rightDF[cols].mean()
                    tempRow[cols] = avg
            newDF.loc[len(newDF)] = tempRow


    #Create a dictionary to find average GDI of patient day of surgery.
    patientLeftGDI = {}
    patientRightGDI = {}

    #Get the Avg GDI for Left Side
    for p in patients:
        pFrame = newDF.loc[newDF['DeIdentifyNumber'] == p]
        pFrame = pFrame.loc[pFrame['StudyType_Pre-op'] == 1]
        pFrame = pFrame.loc[pFrame['MotionParams.Side_Left'] == 1]
        if(pFrame.shape[0] > 0):
            gdiList = pFrame['GDI'].tolist()
            if (len(gdiList) != 0):
                avgGdi = sum(gdiList)/len(gdiList)
            patientLeftGDI[p] = avgGdi

    #Get the Avg GDI for the Right Side
    for p in patients:
        pFrame = newDF.loc[newDF['DeIdentifyNumber'] == p]
        pFrame = pFrame.loc[pFrame['StudyType_Pre-op'] == 1]
        pFrame = pFrame.loc[pFrame['MotionParams.Side_Left'] == 0]
        if(pFrame.shape[0] > 0):
            gdiList = pFrame['GDI'].tolist()
            if (len(gdiList) != 0):
                avgGdi = sum(gdiList)/len(gdiList)
            patientRightGDI[p] = avgGdi

    #Calculate GDI Differential for row.
    for i, row in newDF.iterrows():
        row['Gdi Differential'] = row['GDI'] - row['DeIdentifyNumber']

    doublesx = []
    def gdiDiff(row):
        if (row['MotionParams.Side_Left'] == 1):
            if (row['DeIdentifyNumber'] in patientLeftGDI):
                x = row['GDI'] - patientLeftGDI[row['DeIdentifyNumber']]
                if (x != 0 and row['StudyType_Pre-op'] == 1):
                    print(row['DeIdentifyNumber'])
                doublesx.append(x)
                return x

        if (row['MotionParams.Side_Right'] == 1):
            if (row['DeIdentifyNumber'] in patientRightGDI):
                x = row['GDI'] - patientRightGDI[row['DeIdentifyNumber']]
                if (x != 0 and row['StudyType_Pre-op'] == 1):
                    print(row['DeIdentifyNumber'])
                doublesx.append(x)
                return x

        else:
            return 'n/a'

    #If a patient had a surgery, calculate the days from surgery for each row. Otherwise, return n/a
    def timeDiff(row):
        if (sx[row['DeIdentifyNumber']].shape[0] > 0):
            x = pd.to_datetime(row['TestDate']) - pd.to_datetime(sx[row['DeIdentifyNumber']][0])
            return x.days
        else:
            return 'n/a'

    # Remove patients with multiple surgeries
    for p in doublesx:
        newDF = newDF[newDF['DeIdentifyNumber'] != p]

    #Apply functions defined above
    newDF['Time From Surgery'] = newDF.apply (lambda row: timeDiff(row),axis=1)
    newDF['GDI Differential'] =  newDF.apply (lambda row: gdiDiff(row), axis=1)

    #Drop patients who don't have surgeries
    newDF = newDF[newDF['Time From Surgery'] != 'n/a']
    newDF = newDF[newDF['GDI Differential'] != 'n/a']

    #Find patients who have a post op and a pre op.
    postWPre = []
    for p in patients:
        if(len(sx[p]) > 0):
            x = newDF[newDF['DeIdentifyNumber'] == p]
            d  = x['StudyType_Post-op']
            if (d.shape[0] > 0):
                postWPre.append(p)

    #Filter so it's only patients pre and post checkups
    newDF = newDF.loc[newDF['DeIdentifyNumber'].isin(postWPre)]
    newDF = newDF.loc[newDF['Time From Surgery'] >= 0]

    plt.scatter(newDF['Time From Surgery'], newDF['GDI Differential'])
    plt.title("Time since Surgery")
    plt.xlabel("Days from Surgery")
    plt.ylabel("GDI Differential")
    plt.axhline()
    plt.show()

main()