import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from datetime import *
from lib.massager import *

massage = massager()
filePath = "~/Documents/Loyola/Research/Shriner/data/CP_HAM.csv"

#Read in relevant columns, convert categorical variables to binary.
df = pd.read_csv(filePath)
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

# print(len(sx))
# print(len(patientGDI))

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
        x = pd.to_datetime(row['TestDate']) - pd.to_datetime(sx[row['DeIdentifyNumber']][0])
        return x.days
    else:
        return 'n/a'

# def movingaverage(interval, window_size):
#     window= np.ones(int(window_size))/float(window_size)
#     return np.convolve(interval, window, 'same')
#
# def runningMeanFast(x, N):
#     return np.convolve(x, np.ones((N,))/N)[(N-1):]
#
# def running_mean(x, n):
#     # cumsum = np.cumsum(np.insert(x, 0, 0))
#     # return (cumsum[N:] - cumsum[:-N]) / N
#     ret = np.cumsum(x)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

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

    avg_mean_df = calculate_running_mean_list(N, dict_data, df, 'Time From Surgery', 'GDI Differential')
    return avg_mean_df


dmDF['Time From Surgery'] = dmDF.apply(lambda row: sd(row),axis=1)
dmDF['GDI Differential'] = dmDF.apply(lambda row: diff(row), axis=1)

dmDF = dmDF[dmDF['Time From Surgery'] != 'n/a']
dmDF = dmDF[dmDF['GDI Differential'] != 'n/a']


sxBTXHams = dmDF.loc[dmDF['SxCode_BTX-Hams'] == 1]
sxHamsRelease = dmDF.loc[dmDF['SxCode_Ham-Release'] == 1]
sxHamsM = dmDF.loc[dmDF['SxCode_Hams-M'] == 1]
sxHamsML = dmDF.loc[dmDF['SxCode_Hams-ML'] == 1]

avg_sxBTXHams = movingDateAvg(sxBTXHams, 20)
avg_sxBTXHams = avg_sxBTXHams.sort_values(by= 'Time From Surgery', ascending=True)


# cumMovAvg(sxBTXHams)

# getYearAvg(sxBTXHams)
# columns = ['GDI', 'newIndex']
sxBTXHams_Vector = pd.DataFrame()
sxBTXHams_Vector = sxBTXHams_Vector.append(sxBTXHams['Time From Surgery'])
sxBTXHams_Vector = sxBTXHams_Vector.append(sxBTXHams['GDI Differential'])
sxBTXHams_Vector = sxBTXHams_Vector.transpose()
sxBTXHams_Vector = sxBTXHams_Vector.set_index(sxBTXHams_Vector['Time From Surgery'])

# print(sxBTXHams_Vector)

plt.figure(1)

# plt.plot([1,2,3])
# plt.subplot(221)
plt.scatter(sxBTXHams['Time From Surgery'], sxBTXHams['GDI Differential'])
plt.scatter(avg_sxBTXHams['Time From Surgery'], avg_sxBTXHams['Running_Mean_Avg'])
plt.plot(avg_sxBTXHams['Time From Surgery'], avg_sxBTXHams['Running_Mean_Avg'], marker='o', color='r')
plt.title("Time since Surgery (BTX-Hams)")
plt.xlabel("Days from Surgery")
plt.ylabel("GDI Differential")
plt.axhline()

# plt.subplot(222)
# plt.scatter(sxHamsRelease['Time From Surgery'], sxHamsRelease['GDI Differential'])
# plt.plot(pd.rolling_mean(sxHamsRelease['GDI Differential'], window=2, center=False), "r")
# plt.title("Time from First Surgery (Hams Release)")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()
#
# plt.subplot(223)
# plt.scatter(sxHamsM['Time From Surgery'], sxHamsM['GDI Differential'])
# plt.plot(pd.rolling_mean(sxHamsM['GDI Differential'], window=2), "r")
# plt.title("Time from First Surgery (Hams M")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()
#
# plt.subplot(224)
# plt.scatter(sxHamsML['Time From Surgery'], sxHamsML['GDI Differential'])
# plt.plot(pd.rolling_mean(sxHamsML['GDI Differential'], window=5), "r")
# plt.title("Time from First Surgery (Hams ML")
# plt.xlabel("Days from Surgery")
# plt.ylabel("GDI Differential")
# plt.axhline()
plt.show()