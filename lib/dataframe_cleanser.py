"""
Creates graphs for any variable that is needed,
slicing where things are neccesary.

Author: @Anirrudh Krishnan; Github: @anirrudh
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib as plt
from plotly.graph_objs import Scatter, Figure, Layout

def getfile():
    print('Hello, welcome to the cleaning script.')
    filename = input("Please drag and drop your .csv file here, or enter filepath:\n")
    df = pd.read_csv(filename)
    return df

#Get the column values
def getColumnValues(df):
    return (list(df.columns.values))

#Get unique values for all of the columns in the dataset
def getUniqueValues(df, col):
    return df[col].unique()

#Delete the data that we do not want, based on number.
def sliceoffdata(df, col, int):
    df = df.loc[df[col] == int]
    return df

#Recast the column values to integers for faster
#calculation later
def recastValues(df, col, col_val, recasting_var):
    df[col] = df[col].map({col_val: recasting_var})
    # return df

#Collapse dataframe and return the dict to the function,
#then making it easy to get append this to a list to construct
#the final dataframe.
def returnDict(df):
    """
    Calculate the means of all the columns, then setting these to variables and creating
    something that is shaped the same way.
    """
    lst = ['WalkingSpeedMetersPerSec', 'StepLengthM', 'FootOffPct', 'PelvicTiltMax',\
     'PelvicTiltMin', 'HipFlexionMin', 'HipFlexionMax', 'KneeFlexionMin', 'KneeFlexionPeak',\
     'KneeFlexionAtIC', 'AnkleDorsiPlantarPeakPF', 'AnkleDorsiPlantarPeakDF', 'GDI', 'Happiness',\
     'GlobalFunc2', 'SportsPhysFunc2', 'XfersBasicMobility']
    lst_means = []
    for items in lst:
        mean = df[items].mean()
        lst_means.append(mean)
    dictionary = dict(zip(lst, lst_means))
    lst_nominal = ['DeIdentifyNumber', 'Gender', 'PrimaryDiagnosis', 'Procedures.Side', 'SxCode',\
                   'StudyType', 'TestDate', 'AgeTest(yrs)', 'Trial_GcdFile', 'MotionParams.Side']
    # print(dictionary)
    lst_given = []
    for item in lst_nominal:
        data = df[item].unique()
        try:
            lst_given.append(data[0])
        except:
            lst_given.append(np.nan)
    # print(lst_given)
    dictionary2 = dict(zip(lst_nominal, lst_given))
    dict_final = dict(dictionary2, **dictionary)
    return dict_final
    #print(len(lst), len(lst_means))

#Collapse the data for the rows in each df, if there are more than one
#by finding the mean.
def collapseRows(df, col):
    df = df[pd.notnull(df[col])]
    lst = getUniqueValues(df, col)
    lst = [x for x in lst if ~np.isnan(x)]
    lst = [float(i) for i in lst]
    df[col] = df[col].astype(int)
    # print(lst)
    # print(df.dtypes)
    # df = recastValues(df, 'MotionParams.Side', 'L','Left')
    df['MotionParams.Side'] = df['MotionParams.Side'].map({'Left': 'L', 'Right':'R'})
    # df = recastValues(df, 'MotionParams.Side', )
    final_frame = []
    for pid in lst:
        #create temporary dataframes to collapse the
        #rows into one row, with afflicted sides.
        tmp_df1 = df.loc[df[col] == pid]
        dates = tmp_df1['TestDate'].unique()
        for date in dates:
            tmp_df = tmp_df1.loc[tmp_df1['TestDate'] == date]
            tmp_df_LL = tmp_df.loc[(tmp_df['Procedures.Side'] == 'L') & (tmp_df['MotionParams.Side'] == 'L')]
            tmp_df_LR = tmp_df.loc[(tmp_df['Procedures.Side'] == 'L') & (tmp_df['MotionParams.Side'] == 'R')]
            tmp_df_RR = tmp_df.loc[(tmp_df['Procedures.Side'] == 'R') & (tmp_df['MotionParams.Side'] == 'R')]
            tmp_df_RL = tmp_df.loc[(tmp_df['Procedures.Side'] == 'R') & (tmp_df['MotionParams.Side'] == 'L')]
            leftleft = returnDict(tmp_df_LL)
            rightright = returnDict(tmp_df_RR)
            leftright = returnDict(tmp_df_LR)
            rightleft = returnDict(tmp_df_RL)
            final_frame.append(leftleft)
            final_frame.append(leftright)
            final_frame.append(rightleft)
            final_frame.append(rightright)


    #print(final_frame)
    df2 = pd.DataFrame(final_frame)
    df2 = df2[pd.notnull(df2[col])]
    return df2

#main function to calculate and return a cleansed dataframe
def main():
    df = getfile()
    df = collapseRows(df, 'DeIdentifyNumber')
    return df

