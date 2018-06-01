'''
import data is first step.

'''
import pandas as pd
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
dfcat = df.iloc[:, [1,3,4,5,9]].copy()
catnames = list(dfcat)
dmDF = pd.get_dummies(df, columns= catnames)
dmDF = dmDF.dropna()

#print(list(dmDF))

df1 = dmDF.loc[dmDF['SxCode_Hams-M'] == 1]
df2 = dmDF.loc[dmDF['SxCode_Ham-Release'] == 1]
df3 = dmDF.loc[dmDF['SxCode_BTX-Hams'] == 1]
df4 = dmDF.loc[dmDF['SxCode_Hams-ML'] == 1]

plt.scatter(df1['AgeTest(yrs)'], df1['GDI'])
plt.title("GDI vs Age (Sx - Hams-M)")
plt.xlabel("Age (years)")
plt.ylabel("GDI")
plt.show()

plt.scatter(df2['AgeTest(yrs)'], df2['GDI'])
plt.title("GDI vs Age (Sx - Hams-Release)")
plt.xlabel("Age (years)")
plt.ylabel("GDI")
plt.show()

plt.scatter(df3['AgeTest(yrs)'], df3['GDI'])
plt.title("GDI vs Age (Sx - BTX-Hams)")
plt.xlabel("Age (years)")
plt.ylabel("GDI")
plt.show()

plt.scatter(df4['AgeTest(yrs)'], df4['GDI'])
plt.title("GDI vs Age (Sx - Hams-ML)")
plt.xlabel("Age (years)")
plt.ylabel("GDI")
plt.show()

