"""
This script produces a couple of graphs of the
running mean where it runs the data and creates
beautiful visuals.

@Authors: Anirrudh and Spencer
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

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def getfilepath():
    """
    Get the filepath of the csv file that needs to be loaded.
    :return:
    """
    filepath = input('Please Drag and Drop your CSV File here: ')
    df = pd.read_excel(filepath)
    return df

def getYearAvg(df):
    unique_vals = df['Time From Surgery'].unique()
    unique_vals.sort()
    for x in unique_vals:
        print(x)

def myround(x, base=5):
    f = int(base * round(float(x) / base))
    if f == 0:
        f += 5
    else:
        f = f
    return f

def cumMovAvg(df):
    """
    This will be used to figure out
    the days that we have had and the averages
    that exist therein them, so that we can connect them
    and stitch them together.
    """

    # Calculate Step Length via binning process
    df_time = df['Time From Surgery'].sort_values(ascending=True)
    # print(df_time)
    least_time = df_time.iloc[0]
    most_time = df_time.iloc[len(df) - 1]
    print(least_time, most_time)

    # Use Max - Min Averaging to Figure out N values; the more reccurent they are the smoother they will be
    N = (least_time + most_time) / len(df_time)
    print(N)
    N = myround(N)
    print('New N:', N)

    # Now that N is set, divide the dataframe into the 'N' Values. then, we will go
    # Ahead and figure out how many rows are left at the end, which will be averaged
    # In an ungraceful way

    last_set = len(df) % N
    how_many_extra = len(df) / N
    print('Extra set of rows', how_many_extra)
    print('Last set of rows:', last_set)

    # Write a 'For' loop to split data into 'N' sizes,
    # Then average the columns for that 'N' data,
    # And write to a new df containing the averages and
    # the x and y values.

    i = 0
    row_start = 0
    while i < len(df) - last_set:
        tmp_df = df.iloc[row_start:N]
        row_start = row_start + N
        N += N
        i = i + N
        # print(tmp_df)

def calculate_running_mean_list(window_size, dict_of_data, original_df, col_to_search, col_to_sum):

    print(dict_of_data)
    patient_dates = []
    average_values = []
    for k,v in dict_of_data.items():
        tmp_list = []
        tmp_list.append(v)
        patient_dates.append(k)
        for item in tmp_list:
            tmp_add_array = []
            for item in item:
                rows = original_df.loc[original_df[col_to_search] == item]
                print('======================')
                print(rows[col_to_sum])
                print(item)
                print(rows[col_to_sum].mean())
                print('======================')
                tmp_add_array.append(rows[col_to_sum].mean())
            average_values.append(np.sum(tmp_add_array))
    a = np.array(average_values)
    b = np.empty(len(average_values))

    b.fill(window_size)
    divided_array = np.divide(a, b)
    scaled_divided_array = np.multiply(divided_array, 10)

    try:
        len(dict_of_data) == len(average_values)

    except:
        print("Something is messed up with the data. Please re-check.")

    print(scaled_divided_array)
    print(patient_dates)

    scaled_df_avg = pd.DataFrame({'Time From Surgery': patient_dates, 'Running_Mean_Avg': scaled_divided_array})
    return scaled_df_avg

def movingDateAvg(df, N):
    """
    :param df: The dataframe that we want to run a cumulative mean on
    :param N: The size of the window (in one direction, so make this the TOTAL length of the window)
    :return: Dataframe that will be graphed ontop of the graphs that are generated.
    """
    # Get the total window size Needed for the dataset "Moving Average" by calculating lower
    # and upper bounds
    print('Window Size =', N)
    window_size = int(N / 2)
    lower_bound = window_size * -1
    upper_bound = window_size
    print('Upper Bound:', upper_bound, '\n', 'Lower Bound', lower_bound)

    # Check that the lower and upper bounds are valid
    try:
        lower_bound < upper_bound
    except:
        print("Your boundary conditions are incorrect.")


    # Create a massager object and use the method to clean the
    # Values by making everything within it a DateTime Object
    # df = massage.typeCastToDateTime(df, 'Time From Surgery')
    unique_dates = df['Time From Surgery'].unique()

    # Create a dataframe to store the dateranges and values needed
    columns = ['Time From Surgery', 'UpperDate', 'LowerDate']
    df_of_ranges = pd.DataFrame(columns=columns)
    for every_date in unique_dates:
        lower_date = every_date + lower_bound
        upper_date = every_date + upper_bound
        df_of_ranges = df_of_ranges.append({'Time From Surgery': every_date,'UpperDate': upper_date, 'LowerDate' : lower_date}, ignore_index=True)

    print(list(df_of_ranges))
    print(df_of_ranges.index)

    # for index, row in df_of_ranges.iteritems():
    #     print(row['TestDate'])
    total_frame = []
    for ix in df_of_ranges.index:
        df2 = []
        upper_bound_date = df_of_ranges.loc[ix]['UpperDate']
        lower_bound_date = df_of_ranges.loc[ix]['LowerDate']

        for date in unique_dates:
            if lower_bound_date < date < upper_bound_date:
                df2.append(date)


        total_frame.append(df2)
    # print(df_of_ranges.loc[1]['UpperDate'], df_of_ranges.loc[1]['LowerDate'])
    # print(total_frame[1])


    # Create a Dict that has the key set to date, and the value set to the date ranges
    dict_data = {}
    i = 0
    while i < len(total_frame):
        dict_data.update({df_of_ranges['Time From Surgery'].iloc[i] : total_frame[i]})
        i += 1

    """
    Now that we have that the "range" of the value for each date, let's use the dict to locate the
    values that are needed to do the computation for each point, returning a set of the data points and
    averaging those values. 
    N : Window Size 
    """

    avg_mean_df = calculate_running_mean_list(N, dict_data, df, 'Time From Surgery', 'HipFlexionMax')
    return avg_mean_df

def main():
    #Get filepath, drop GDI's under 20 as cleansing measure, create new columns from categorical variables.
    df = getfilepath()
    print(list(df))
    df = df[df['GDI'] > 20]
    cat = ['Gender', 'Procedures.Side', 'SxCode', 'StudyType', 'MotionParams.Side']
    dmDF = pd.get_dummies(df, columns= cat)
    dmDF = dmDF.drop(['Comment'], axis=1)
    dmDF = dmDF.drop(['Trial_GcdFile'], axis=1)
    dmDF = dmDF.dropna()
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


    #Create a dictionary to find average GDI of patient day of surgery, separated by side.
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

    #Remove patients with multiple surgeries
    for p in doublesx:
        newDF = newDF[newDF['DeIdentifyNumber'] != p]

    #Apply functions defined above to calculate Time and GDI differential graphs
    newDF['Time From Surgery'] = newDF.apply (lambda row: timeDiff(row),axis=1)
    newDF['GDI Differential'] =  newDF.apply (lambda row: gdiDiff(row), axis=1)

    #Drop patients who don't have surgeries
    newDF = newDF[newDF['Time From Surgery'] != 'n/a']
    newDF = newDF[newDF['GDI Differential'] != 'n/a']

    #Graph results based on surgery
    sxBTXHams = newDF.loc[newDF['SxCode_BTX-Hams'] == 1]
    sxHamsRelease = newDF.loc[newDF['SxCode_Ham-Release'] == 1]
    sxHamsM = newDF.loc[newDF['SxCode_Hams-M'] == 1]
    sxHamsML = newDF.loc[newDF['SxCode_Hams-ML'] == 1]

    avg_sxBTXHams = movingDateAvg(sxBTXHams, 10)
    avg_sxHamsRelease = movingDateAvg(sxHamsRelease, 10)
    avg_sxHamsM = movingDateAvg(sxHamsM, 10)
    avg_sxHamsML = movingDateAvg(sxHamsML, 10)

    avg_sxBTXHams = avg_sxBTXHams.sort_values(by= 'Time From Surgery', ascending=True)
    avg_sxHamsRelease = avg_sxHamsRelease.sort_values(by= 'Time From Surgery', ascending=True)
    avg_sxHamsM = avg_sxHamsM.sort_values(by= 'Time From Surgery', ascending=True)
    avg_sxHamsML = avg_sxHamsML.sort_values(by= 'Time From Surgery', ascending=True)

    #Define Figure to hold all our plots
    plt.figure(1)

    #First plot to graph BTX and running means
    plt.plot([1,2,3])
    plt.subplot(221)
    plt.scatter(sxBTXHams['Time From Surgery'], sxBTXHams['HipFlexionMax'])
    plt.scatter(avg_sxBTXHams['Time From Surgery'], avg_sxBTXHams['Running_Mean_Avg'])
    plt.plot(avg_sxBTXHams['Time From Surgery'], avg_sxBTXHams['Running_Mean_Avg'], marker='o', color='r')
    plt.title("Time since Surgery (BTX-Hams)")
    plt.xlabel("Days from Surgery")
    plt.ylabel("GDI Differential")
    plt.axhline()

    #Second plot
    plt.subplot(222)
    plt.scatter(sxHamsRelease['Time From Surgery'], sxHamsRelease['HipFlexionMax'])
    plt.scatter(avg_sxHamsRelease['Time From Surgery'], avg_sxHamsRelease['Running_Mean_Avg'])
    plt.plot(avg_sxHamsRelease['Time From Surgery'], avg_sxHamsRelease['Running_Mean_Avg'], marker='o', color='r')
    plt.title("Time from First Surgery (Hams Release)")
    plt.xlabel("Days from Surgery")
    plt.ylabel("GDI Differential")
    plt.axhline()

    #3rd plot
    plt.subplot(223)
    plt.scatter(sxHamsM['Time From Surgery'], sxHamsM['HipFlexionMax'])
    plt.scatter(avg_sxHamsM['Time From Surgery'], avg_sxHamsM['Running_Mean_Avg'])
    plt.plot(avg_sxHamsM['Time From Surgery'], avg_sxHamsM['Running_Mean_Avg'], marker='o', color='r')
    plt.title("Time from First Surgery (Hams M")
    plt.xlabel("Days from Surgery")
    plt.ylabel("GDI Differential")
    plt.axhline()

    #4th plot
    plt.subplot(224)
    plt.scatter(sxHamsML['Time From Surgery'], sxHamsML['HipFlexionMax'])
    plt.scatter(avg_sxHamsML['Time From Surgery'], avg_sxHamsML['Running_Mean_Avg'])
    plt.plot(avg_sxHamsML['Time From Surgery'], avg_sxHamsML['Running_Mean_Avg'], marker='o', color='r')
    plt.title("Time from First Surgery (Hams ML")
    plt.xlabel("Days from Surgery")
    plt.ylabel("GDI Differential")
    plt.axhline()
    plt.show()

main()