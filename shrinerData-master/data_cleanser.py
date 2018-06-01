"""
@Author: Anirrudh Krishnan

Cleanse and dump data however I feel like doing it.
Taking in user input is a must.

Good for some fun stuff!

"""

import pandas as pd
import numpy as np
import sys, os
import re
from datetime import datetime
from dateutil.parser import parse

#Dictionary based on month values

month_dict = {"Jan":'01', "Feb":'02', "Mar":'03', "Apr":'04', "May":'05', "Jun":'06', "Jul":'07', \
"Aug":'08', "Sep":'09', "Oct":'10', "Nov":'11', "Dec":'12' }

#Define file path, get working dir
path = os.getcwd()

#Load file
def loadCSV(wdir, filename):
    filepath = os.path.join(wdir, filename)
    dir_file = pd.read_csv(filepath)
    return dir_file

#Print each distinct value in any column that is needed
def getColDistData(df, col):
    df[col] = df[col].values
    new_df, indices = np.unique(df[col], return_inverse=True)
    print new_df

#Slice any patient based on any exact value
def sliceByValue(df, col, value):
    return df.loc[df[col] == value]

#Delete any null valued columns in the data
def killNullValues(df, col):
    new_df = df[df[col].notnull()]
    #Check that there are no null values
    getColDistData(new_df, col)
    #return the dataframe
    return new_df

#Get any additional values from Dictionary
def getDictVal(string_key, dict_from):
    try:
        if string_key in dict_from:
            return dict_from.get(string_key)
    except:
        print("Yikes, nothing came up in the dict. Try debugging?")

#Get Total Number of days in the hospital based on last day given data
def getTotalDays(df_in, time_col):
    # Get Last Day and First Day
    lastDay = df_in[time_col].iloc[-1]
    firstDay = df_in[time_col].iloc[0]

    # Replace the values using regex and dict
    # Search Strings
    month_last = re.search('[a-zA-Z]+', lastDay)
    month_first = re.search('[a-zA-Z]+', firstDay)

    # Extract just the month information
    month_last = month_last.group()
    month_first = month_first.group()

    # Replace the value with one from the month_dict
    month_last = getDictVal(month_last, month_dict)
    month_first = getDictVal(month_first, month_dict)

    # Replace the value within the actual string iteslf...
    lastDay = re.sub('[a-zA-Z]+', month_last, lastDay)
    firstDay = re.sub('[a-zA-Z]+', month_first, firstDay)

    # Parse them in as date-time objects, then subtract them.
    lastDay = parse(lastDay)
    firstDay = parse(firstDay)
    lastDay = lastDay.date()
    firstDay = firstDay.date()

    # Calculate Total Days
    print lastDay
    print firstDay
    totalDay = (lastDay - firstDay).days
    return totalDay

#Slice Data based on age
def sliceOnAge():
    print("To be continued!")

#Get average of any column
def getAvgCol(df, col_name):
    avg = np.mean(df[col_name])
    return avg

#Slice data by only returning any patient data
def returnPatientDF(PID):
    patient_df = sliceByValue(df, 'DeIdentifyNumber', PID)

    #Get some data that we need to Calculate
    total_time = getTotalDays(patient_df, 'TestDate')
    avg_gdi = getAvgCol(patient_df, 'GDI')
    avg_happiness = getAvgCol(patient_df, 'Happiness')

    #Write to the dataframe
    patient_df['Total Days of Admit'] = total_time
    patient_df['Avg GDI'] = avg_gdi
    patient_df['Avg Happiness'] = avg_happiness
    
    #Change the column name
    patient_df = patient_df.rename(index=str, columns={'AgeTest(yrs)':'AgeTest'})
    patient_df['Procedures.Side'] = patient_df['Procedures.Side'].map({'R': "Right", 'L': "Left"})

    print patient_df
    return patient_df

#Write dataframe to csv
def data2CSV(df, PID):
    filepath = os.getcwd() + "/Sliced_Data"
    filename = filepath + "/" + str(PID) + ".csv"
    df.to_csv(filename, encoding='utf-8')
    print("Sucessfully wrote the frame to a .csv file!")

#Change column names...for R dataframe


csv_file = raw_input("What is the name of the file that you would like to load?: ")
csv_file = "data/" + csv_file + ".csv"
dirty_file = loadCSV(path, csv_file)
#Kill any null gender values
df = killNullValues(dirty_file, 'Gender')
#Change the column ID to another column by copying values over!
df['DeIdentifyNumber'] = df['DeIdentifyNumber'].astype(int)


def main():
    getColDistData(df, 'DeIdentifyNumber')
    what_PID = input("What is the patient PID you are looking for?: ")
    patient1 = returnPatientDF(what_PID)
    data2CSV(patient1, what_PID)

main()