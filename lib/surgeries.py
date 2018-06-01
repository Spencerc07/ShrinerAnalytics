"""
This file is to slice the patients and all of the surgeries that they have had,
by cleansing this file, then generating a scatterplot using ggplot
"""

import pandas as pd 
import numpy as np
import sys, os
from massager import *

massage = massager()
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
print(path)
#import data
def loadFullData(): 
    # csv_file = input("What is the name of the file that you would like to load?: ")
    csv_file = "CP_HAM"
    csv_file = "/data/" + csv_file + ".csv"
    csv_file = path + csv_file
    full_file = massage.loadCSV(path, csv_file)
    return full_file

def dropNull(df, col):
    df = df[df[col].notnull()]
    return df

def renameSides(df):
    df['Procedures.Side'] = df['Procedures.Side'].replace({'L':0})
    df['Procedures.Side'] = df['Procedures.Side'].replace({'R':1})
    df['MotionParams.Side'] = df['MotionParams.Side'].replace({'Right':1})
    df['MotionParams.Side'] = df['MotionParams.Side'].replace({'Left':0})
    # print(df)
    # for item in df['Procedures.Side'].iteritems(): 
    #     if item == 'L':
#Slice only the side that procedures are done on
#Get one date per patient


a = loadFullData()
a = dropNull(a, 'TestDate')
a = dropNull(a, 'Procedures.Side')
a['TestDate'] = pd.to_datetime(a['TestDate'])
a['DeIdentifyNumber'] = a['DeIdentifyNumber'].astype(int)
# print(a['TestDate'].dtypes)
# print(a['TestDate'])
renameSides(a)

b = a.drop_duplicates(['TestDate'])
# print(b['TestDate'].dtypes)
# print(b['DeIdentifyNumber'].value_counts())
c = {} 
c = b['DeIdentifyNumber'].value_counts()
# print(c)
print(len(b))
for k,v in c.iteritems(): 
    if v > 1: 
        del c[k]

b = b.set_index('DeIdentifyNumber')
# print(c)
# print(b)
print(len(c))

for k,v in c.iteritems():
    b = b.drop(k)
    print(len(b))

print(b)
b = b.reset_index()
d = b['DeIdentifyNumber'].value_counts()
print(d)
# b = b.set_index('DeIdentifyNumber')
#Create a new column for each surgery by patient 
SurgeryVisitNum = []
finalCol = []

df = pd.DataFrame()
for k,v in d.iteritems(): 
    e = b.loc[b['DeIdentifyNumber'] == k]
    e.sort_values(by = ['TestDate'])
    i = 0
    while i < v:
        i += 1
        SurgeryVisitNum.append(i)
    df = df.append(e)

se = pd.Series(SurgeryVisitNum)
df['SurgeryVisNum'] = se.values



filepath = path + "/Sliced_Data"
filename = filepath + "/" + "unique_dates" + ".csv"
df.to_csv(filename, encoding='utf-8')
print("Sucessfully wrote the frame to a .csv file!")