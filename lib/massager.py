"""
@Author: Anirrudh Krishnan

Cleanse and dump data however I feel like doing it.
Taking in user input is a must.

Good for some fun stuff!

This version is using python3

"""

import pandas as pd
import numpy as np
import sys, os
import re
from datetime import datetime
from dateutil.parser import parse

class massager(): 

    #Define file path, get working dir
    @staticmethod
    def loadCSV(wdir, filename):
        """
        Load the csv file by getting file path. All .csv files should exist in 
        /data folder. 
        """
        filepath = os.path.join(wdir, filename)
        dir_file = pd.read_csv(filepath)
        return(dir_file)

    @staticmethod
    def getColDistData(df, col):
        """
        Print unique column data from a specific column. 
        Can use to probe datasets, just change the code here. 
        """
        df[col] = df[col].values
        new_df, indices = np.unique(df[col], return_inverse=True)
        return new_df

    @staticmethod
    def sliceByValue(df, col, value):
        """
        Use mapping method to grab the
        specific patient ID value we are looking for
        """ 
        return df.loc[df[col] == value]

    @classmethod
    def killNullValues(df, col, cls):
        """
        Generate a new dataset that does not
        contain any null values in a specific column. Typically
        run before cleaning specific patient data. 
        """
        new_df = df[df[col].notnull()]
        getColDistData(new_df, col)
        return new_df

    @staticmethod
    def getDictVal(string_key, dict_from):
        """
        Get any values from dictionary, currently only used for 
        getting month and remapping for datetime (Mon Feb 5th '18)
        """
        try:
            if string_key in dict_from:
                return dict_from.get(string_key)
        except:
            print("Yikes, nothing came up in the dict. Try debugging?")

    @staticmethod
    def getTotalDays(df_in, time_col):
        """
        Get the total amount of time recorded in days 
        for the total time spend at the hospital for the patient. 
        """
        month_dict = {"Jan":'01', "Feb":'02', "Mar":'03', "Apr":'04', "May":'05', "Jun":'06', "Jul":'07', \
        "Aug":'08', "Sep":'09', "Oct":'10', "Nov":'11', "Dec":'12' }
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
        # print lastDay
        # print firstDay
        totalDay = (lastDay - firstDay).days
        return totalDay

    @classmethod
    def typeCastToDateTime(cls, df_in, time_col):
        month_dict = {"Jan": '01', "Feb": '02', "Mar": '03', "Apr": '04', "May": '05', "Jun": '06', "Jul": '07', \
                      "Aug": '08', "Sep": '09', "Oct": '10', "Nov": '11', "Dec": '12'}
        for index, i in df_in[time_col].iteritems():
            month = re.search('[a-zA-Z]+', i)
            month_name = month.group()
            month_name = cls.getDictVal(month_name, month_dict)
            j = re.sub('[a-zA-Z]+', month_name, i)
            # print(j)
            # df_in[time_col].replace(i, j)
            k = parse(j)
            k = k.date()
            df_in.loc[df_in[time_col] == i, time_col] = k

        return df_in





    @staticmethod
    def getAvgCol(df, col_name):
        """
        Get mean of any col
        """
        avg = np.mean(df[col_name])
        return avg

    #Write dataframe to csv
    @staticmethod
    def data2CSV(df, PID):
        filepath = os.getcwd() + "/Sliced_Data"
        filename = filepath + "/" + str(PID) + ".csv"
        df.to_csv(filename, encoding='utf-8')
        print("Successfully wrote the frame to a .csv file!")

