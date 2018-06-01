"""
Runs a T-Test of the seperate,
two sided variables. 
For the Purpose of this project, 
we will use the M/F Gender types
and run this T-Test. 
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import scale
import dataframe_cleanser

df =  dataframe_cleanser.main()
df = df.dropna(axis=0, how='any')
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

#Create a dataframe for the variables
df_male = df.loc[df.Gender == 0]
df_female = df.loc[df.Gender == 1]

#Look at the data that we want to run the T-Test on. 
def t_tester(col)
    male = df_male[col]
    male = scale(male)
    female = df_female[col]
    female = scale(female)
    print('T-Test Results for Column', col)
    print('=================================')
    t, p = ttest_ind(male, female, equal_var=False)
    print("ttest_ind:            t = %g  p = %g" % (t, p))