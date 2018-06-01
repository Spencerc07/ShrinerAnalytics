import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from massager import *
import matplotlib.pyplot as plt


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

#Read in relevant columns, convert categorical variables to binary.
df = loadFullData()
dfcat = df.iloc[:, [1,3,4,5,9]].copy()
catnames = list(dfcat)
dmDF = pd.get_dummies(df, columns= catnames)
dmDF = dmDF.drop(['Comment'], axis=1)
dmDF = dmDF.dropna()
#print(dmDF['TestDate'].head())
#Get unique patient ids
patients = dmDF.DeIdentifyNumber.unique()

#print(patients)

#running average gaussian smoothing
#break down by surgery
#y axis should be gdi difference from day of surgery
#so gdi of date of record - gdi day of preop


sx = {}
for p in patients:
    x = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
    x = x[x['StudyType_Pre-op'] != 0]
    sxDate = x['TestDate'].unique()
    sx[p] = sxDate

patientGDI = {}

for p in patients:
    pFrame = dmDF.loc[dmDF['DeIdentifyNumber'] == p]
    pFrame = pFrame[pFrame['StudyType_Pre-op'] != 0]
    if(pFrame.shape[0] > 0):
        pFrame = pFrame[pFrame['Procedures.Side_L'] == pFrame['MotionParams.Side_Left']]
        gdiList = pFrame['GDI'].tolist()
        avgGdi = sum(gdiList)/len(gdiList)
        patientGDI[p] = avgGdi

print(len(sx))
print(len(patientGDI))

for i, row in dmDF.iterrows():
    row['Gdi Differential'] = row['GDI'] - row['DeIdentifyNumber']

def diff(row):
    if(row['DeIdentifyNumber'] in patientGDI):
        x = row['GDI'] - patientGDI[row['DeIdentifyNumber']]
        return x
    else:
        return 'n/a'

def sd (row):
    if (sx[row['DeIdentifyNumber']].shape[0] > 0):
        # print(pd.to_datetime(sx[row['DeIdentifyNumber']][0]))
        x = pd.to_datetime(row['TestDate']) - pd.to_datetime(sx[row['DeIdentifyNumber']][0])
        print(x.days)
        return x.days
    else:
        return 'n/a'

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

dmDF['Time From Surgery'] = dmDF.apply (lambda row: sd(row),axis=1)
dmDF['GDI Differential'] =  dmDF.apply (lambda row: diff(row), axis=1)

dmDF = dmDF[dmDF['Time From Surgery'] != 'n/a']
dmDF = dmDF[dmDF['GDI Differential'] != 'n/a']
# dmDF['TotalDays'] = massage.


# xlim(0,1000)
# xlabel("Months since Jan 1749.")
# ylabel("No. of Sun spots")
# grid(True)

sxBTXHams = dmDF.loc[dmDF['SxCode_BTX-Hams'] == 1]
sxHamsRelease = dmDF.loc[dmDF['SxCode_Ham-Release'] == 1]
sxHamsM = dmDF.loc[dmDF['SxCode_Hams-M'] == 1]
sxHamsML = dmDF.loc[dmDF['SxCode_Hams-ML'] == 1]

new_df = sxBTXHams.append(sxHamsRelease)
new_df = new_df.append(sxHamsM)
new_df = new_df.append(sxHamsML)

new_df = new_df[new_df['Time From Surgery'] != 0]
y_av = movingaverage(dmDF['GDI Differential'], len(dmDF['GDI Differential']) % 40)
plt.plot(y_av)
# plt.figure(1)
plt.show()
# plt.plot([1,2,3])
# plt.subplot(221)
# plt.scatter(sxBTXHams['Time From Surgery'], sxBTXHams['GDI Differential'])
# plt.title("Time since Surgery (BTX-Hams)")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()

# plt.subplot(222)
# plt.scatter(sxHamsRelease['Time From Surgery'], sxHamsRelease['GDI Differential'])
# plt.title("Time from First Surgery (Hams Release)")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()

# plt.subplot(223)
# plt.scatter(sxHamsM['Time From Surgery'], sxHamsM['GDI Differential'])
# plt.title("Time from First Surgery (Hams M")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()

# plt.subplot(224)
# plt.scatter(sxHamsML['Time From Surgery'], sxHamsML['GDI Differential'])
# plt.title("Time from First Surgery (Hams ML")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()
# plt.show()

"""
writer = pd.ExcelWriter('output.xlsx')
dmDF.to_excel(writer,'Sheet1')
writer.save()
"""