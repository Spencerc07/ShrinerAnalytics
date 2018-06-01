import pandas as pd
import datetime
import numpy as np
import scipy
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

filePath = "/Users/spencer/Documents/HW/IS/Shriner.xlsx"

#Read in relevant columns, convert categorical variables to binary.
df = pd.read_excel(filePath)
df = df[df['GDI'] > 20]
dfcat = df.iloc[:, [1,3,4,5,9]].copy()
catnames = list(dfcat)
dmDF = pd.get_dummies(df, columns= catnames)
dmDF = dmDF.drop(['Comment'], axis=1)
dmDF = dmDF.dropna()#
#dmDF['TestDate'] = dmDF['TestDate'].dt.date.datestrftime("%Y-%m-%d")
drop = [94850, 46577, 58885, 70712, 55181, 16273, 98666]
for d in drop:
   dmDF = dmDF[dmDF['DeIdentifyNumber'] != d]
#Get unique patient ids
patients = dmDF.DeIdentifyNumber.unique()
sx = {}
for p in patients:
    x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
    x = x[x['StudyType_Pre-op'] != 0]
    sxDate = x['TestDate'].unique()
    sx[p] = sxDate

vis = []

for p in patients:
    x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
    vis.append(x['TestDate'].unique())

avCols = ['WalkingSpeedMetersPerSec', 'StepLengthM', 'FootOffPct', 'PelvicTiltMax', 'PelvicTiltMin', 'HipFlexionMin',
          'KneeFlexionMin', 'KneeFlexionPeak', 'KneeFlexionAtIC', 'AnkleDorsiPlantarPeakPF', 'GDI']

newDF = pd.DataFrame(columns=list(dmDF))

for p in patients:
    x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
    visits = x['TestDate'].unique()

    for v in visits:
        #Filter df to only have first date
        a = x[x['TestDate'] == v]

        l = a[a['MotionParams.Side_Left'] == 1]

        if (l.shape[0]%3 != 0 or l.shape[0] == 0):
            break

        else:
            temprow = l.iloc[0]

            for cols in avCols:
                avg = l[cols].mean()
                temprow[cols] = avg
        newDF.loc[len(newDF)] = temprow

for p in patients:
    x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
    visits = x['TestDate'].unique()

    for v in visits:
        #Filter df to only have first date
        a = x[x['TestDate'] == v]

        l = a[a['MotionParams.Side_Right'] == 1]

        if (l.shape[0]%3 != 0 or l.shape[0] == 0):
            break

        else:
            temprow = l.iloc[0]

            for cols in avCols:
                avg = l[cols].mean()
                temprow[cols] = avg
        newDF.loc[len(newDF)] = temprow

#Create a dictionary to find average GDI of patient day of surgery.
patientLeftGDI = {}
patientRightGDI = {}

for p in patients:
    pFrame = newDF.loc[newDF['DeIdentifyNumber'] == p]
    pFrame = pFrame.loc[pFrame['StudyType_Pre-op'] == 1]
    pFrame = pFrame.loc[pFrame['MotionParams.Side_Left'] == 1]
    if(pFrame.shape[0] > 0):
        gdiList = pFrame['GDI'].tolist()
        if (len(gdiList) != 0):
            avgGdi = sum(gdiList)/len(gdiList)
        patientLeftGDI[p] = avgGdi

for p in patients:
    pFrame = newDF.loc[newDF['DeIdentifyNumber'] == p]
    pFrame = pFrame.loc[pFrame['StudyType_Pre-op'] == 1]
    pFrame = pFrame.loc[pFrame['MotionParams.Side_Left'] == 0]
    if(pFrame.shape[0] > 0):
        gdiList = pFrame['GDI'].tolist()
        if (len(gdiList) != 0):
            avgGdi = sum(gdiList)/len(gdiList)
        patientRightGDI[p] = avgGdi


for i, row in newDF.iterrows():
    row['Gdi Differential'] = row['GDI'] - row['DeIdentifyNumber']

doublesx = []
def diff(row):
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
                print (row['DeIdentifyNumber'])
            doublesx.append(x)
            return x

    else:
        return 'n/a'

def sd (row):
    if (sx[row['DeIdentifyNumber']].shape[0] > 0):
        x = row['TestDate'] - sx[row['DeIdentifyNumber']][0]
        return x.days
    else:
        return 'n/a'

test = []
for p in patients:
    x = newDF.loc[newDF['DeIdentifyNumber'] == p]
    if (sum(x['StudyType_Pre-op']) > 2):
        test.append(x['DeIdentifyNumber'].iloc[0])

print(test)
print(doublesx)


newDF['Time From Surgery'] = newDF.apply (lambda row: sd(row),axis=1)
newDF['GDI Differential'] =  newDF.apply (lambda row: diff(row), axis=1)

newDF = newDF[newDF['Time From Surgery'] != 'n/a']
newDF = newDF[newDF['GDI Differential'] != 'n/a']


"""

sxBTXHams = newDF.loc[newDF['SxCode_BTX-Hams'] == 1]
sxHamsRelease = newDF.loc[newDF['SxCode_Ham-Release'] == 1]
sxHamsM = newDF.loc[newDF['SxCode_Hams-M'] == 1]
sxHamsML = newDF.loc[newDF['SxCode_Hams-ML'] == 1]

plt.figure(1)

plt.plot([1,2,3])
plt.subplot(221)
plt.scatter(sxBTXHams['Time From Surgery'], sxBTXHams['GDI Differential'])
plt.title("Time since Surgery (BTX-Hams)")
plt.xlabel("Days from Surgery")
plt.ylabel("GDI Differential")
plt.axhline()

plt.subplot(222)
plt.scatter(sxHamsRelease['Time From Surgery'], sxHamsRelease['GDI Differential'])
plt.title("Time from First Surgery (Hams Release)")
plt.xlabel("Days from Surgery")
plt.ylabel("GDI Differential")
plt.axhline()

plt.subplot(223)
plt.scatter(sxHamsM['Time From Surgery'], sxHamsM['GDI Differential'])
plt.title("Time from First Surgery (Hams M")
plt.xlabel("Days from Surgery")
plt.ylabel("GDI Differential")
plt.axhline()

plt.subplot(224)
plt.scatter(sxHamsML['Time From Surgery'], sxHamsML['GDI Differential'])
plt.title("Time from First Surgery (Hams ML")
plt.xlabel("Days from Surgery")
plt.ylabel("GDI Differential")
plt.axhline()
plt.show()


writer = pd.ExcelWriter('newoutput.xlsx')
newDF.to_excel(writer,'Sheet1')
writer.save()
"""