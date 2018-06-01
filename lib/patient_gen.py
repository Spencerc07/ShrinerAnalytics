"""
The point of this class is to seperate 
the patient data class and return a clean 
dataframe sliced by patient. 
"""

import sys
import os
from massager import *
import pandas as pd 
import numpy as np 

massage = massager()
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

class Patient(): 
    """
    This class will generate a patient object 
    Containing a dataframe that will pass off 
    the data to a .csv file and/or R. 
    """

    #Create an object from massager to massage data. 
    
    @staticmethod
    def loadFullData(): 
        csv_file = input("What is the name of the file that you would like to load?: ")
        csv_file = "/data/" + csv_file + ".csv"
        csv_file = path + csv_file
        full_file = massage.loadCSV(path, csv_file)
        return full_file
      
    @classmethod
    def cleanDirtyData(cls, col_to_clean, col_to_typecast):
        full_file = cls.loadFullData()
        df = massage.killNullValues(full_file, col_to_clean)
        df[col_to_typecast] = df[col_to_typecast].astype(int)
        return df

    @classmethod
    def returnPatientDF(PID, df, cls):
        """
        Return a dataframe based on the PID by performing all the computations that are necessary. 
        """
        
        patient_df = massage.sliceByValue(df, 'DeIdentifyNumber', PID)

        #Get some data that we need to Calculate
        total_time = massage.getTotalDays(patient_df, 'TestDate')
        avg_gdi = massage.getAvgCol(patient_df, 'GDI')
        avg_happiness = massage.getAvgCol(patient_df, 'Happiness')

        #Write to the dataframe
        patient_df['Total Days of Admit'] = total_time
        patient_df['Avg GDI'] = avg_gdi
        patient_df['Avg Happiness'] = avg_happiness
        
        #Change the column name
        patient_df = patient_df.rename(index=str, columns={'AgeTest(yrs)':'AgeTest'})
        patient_df['Procedures.Side'] = patient_df['Procedures.Side'].map({'R': "Right", 'L': "Left"})

        return patient_df
    
    @classmethod
    def main(cls):
        df = cls.cleanDirtyData('Gender', 'DeIdentifyNumber')
        massage.getColDistData(df, 'DeIdentifyNumber')
        what_PID = input("What is the patient PID you are looking for?: ")
        patientdf = cls.returnPatientDF(df, what_PID)
        return patientdf

a = Patient()
f = a.main()