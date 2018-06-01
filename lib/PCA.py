"""
Get Dataframe from the dataframe cleaning script,
then run PCA on it to find the significant features.
"""

#Import Code
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import plotly.offline as py
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
# %matplotlib inline
import dataframe_cleanser as dataframe_cleanser

#Get DataFrame
df = dataframe_cleanser.main()

#Drop any and all null values for PCA
df = df.dropna(axis=0, how='any')

"""
Here, we are going to split off into
two different vairables that will be used as axis vairbales.
More can be added when there is a need to.
"""
#Extract data to look at the column variables.
pid = pd.DataFrame(df[['DeIdentifyNumber']])

#I'll go ahead and recast the variable to know which is M and which is F
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

print(list(df.columns.values))
gender = pd.DataFrame(df[['Gender']])

#Create Dataframe with the only variables that matter - the values.
df_values = df[['AnkleDorsiPlantarPeakDF', 'AnkleDorsiPlantarPeakPF', 'FootOffPct', \
                'GDI','GlobalFunc2', 'Happiness', 'HipFlexionMax', 'HipFlexionMin',\
                'KneeFlexionAtIC', 'KneeFlexionMin', 'KneeFlexionPeak','HipFlexionMin',\
                'KneeFlexionAtIC','KneeFlexionMin', 'KneeFlexionPeak','SportsPhysFunc2',\
                'StepLengthM','WalkingSpeedMetersPerSec', 'XfersBasicMobility']]
#Drop any unneeded values that are null. These will not be used in the dataest for scaling.
#The only other option is to set these to 0, but that will screw with the data.
df_values = df_values.dropna(axis=0, how='any')

#numpy array conversion
x = df_values.values

#scale the new matrix values using in built scaler. If this doesn't look good,
#then a max min scale can also be accomplished.
x = scale(x)
components = len(list(df_values.columns.values))

#Run PCA on the maximum amount of components to reduce the dataest.
pca = PCA(n_components=components)
pca.fit(x)
var = pca.explained_variance_ratio_
#Cumulative Variance explains
variance=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print('Variances:\n', variance)

#Finding the covariance and the eigenvalues
df_mat = np.asmatrix(x)
sigma = np.cov(df_mat.T)

#Get the values and sort them.
eigVals, eigVec = np.linalg.eig(sigma)
sorted_index = eigVals.argsort()[::-1]
eigVals = eigVals[sorted_index]
eigVec = eigVec[:,sorted_index]

#Get the first 2 values as they are the most
#significant in the dataset.
eigVec = eigVec[:,:2]
# print(eigVec)
#Dot Product of the matrix with the eigenVector to get the significant features
#out of the dataset.
transformed = df_mat.dot(eigVec)



"""
Get the values for the data and store them in a bar graph to see
visually. 

Taken from Plotly: https://plot.ly/ipython-notebooks/principal-component-analysis/
"""
def checkValuable(eigVals):
    tot = sum(eigVals)
    tot = tot.real
    var_exp = [(i / tot)*100 for i in eigVals]
    var_exp = np.asarray(var_exp)
    var_exp = var_exp.real
    cum_var_exp = np.cumsum(var_exp)
    cum_var_exp = np.asarray(cum_var_exp)
    cum_var_exp = cum_var_exp.real

    trace1 = Bar(
            x=['PC %s' %i for i in range(1,5)],
            y=var_exp,
            showlegend=False)

    trace2 = Scatter(
            x=['PC %s' %i for i in range(1,5)],
            y=cum_var_exp,
            name='cumulative explained variance')

    data = Data([trace1, trace2])

    layout=Layout(
            yaxis=YAxis(title='Explained variance in percent'),
            title='Explained variance by different principal components')

    fig = Figure(data=data, layout=layout)
    py.plot(fig)

def PCAPlt(df, label):
    final_df = np.hstack((df, label))
    final_df = pd.DataFrame(final_df)
    final_df.columns = ['x','y','label']
    groups = final_df.groupby('label')
    figure, axes = plt.subplots()

    axes.margins(0.05)
    for name, group in groups:
        axes.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
        axes.set_title("PCA on CP_HAM")
        axes.legend()
        p1 = round(variance[0],2)
        p1 = str(p1)
        p2 = round(variance[1], 2)
        p2 = str(p2)
    plt.xlabel(("Principal Component 1 %s") % (p1))
    plt.ylabel(("Principal Component 2 %s") % (p2))
    plt.show()

checkValuable(eigVals)
PCAPlt(transformed, gender)
PCAPlt(transformed, pid)