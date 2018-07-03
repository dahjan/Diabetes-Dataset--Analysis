import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


"""
The dataset contains many values which do not make any sense.
As an example, a blood pressure that is >1000 or <=0 is unrealistic!
"""
def clean_data(df):
    return df[ (df.BloodPressure < 1000)
                & (df.BloodPressure > 0)
                & (df.SkinThickness > 0)
                & (df.Glucose > 0)
                & (df.BMI > 0) 
                & (df.Insulin > 0) ]
    

def plot_3D(df, label1, label2, title1, title2):
    #---- Parameters which are the same for both figures
    fig = plt.figure(figsize=(20,8))
    x = np.array(df['Age'])
    y = np.array(df['BloodPressure'])
    z = np.array(df['BMI'])

    #---- First subplot
    ax = fig.add_subplot(121, projection='3d')

    ax.scatter(x, y, z, marker="s", c=label1, s=25, cmap="RdBu")
    ax.set_title("%s" % title1, fontsize = 16)
    ax.set_xlabel('Age')
    ax.set_ylabel('BloodPressure')
    ax.set_zlabel('BMI')
    ax.view_init(15) # rotation of the 3D plot

    #---- Second subplot
    ax = fig.add_subplot(122, projection='3d')

    ax.scatter(x, y, z, marker="s", c=label2, s=25, cmap="RdBu")    
    ax.set_title("%s" % title2, fontsize = 16)
    ax.set_xlabel('Age')
    ax.set_ylabel('BloodPressure')
    ax.set_zlabel('BMI')
    ax.view_init(15) # rotation of the 3D plot

    plt.show()


def plot_TSNE(df, label1, label2, title1, title2):
    fig, ax = plt.subplots(2, sharex=True, figsize=(8,10))

    ax[0].scatter(df[:,0], df[:,1], c = label1, cmap="RdBu")
    ax[0].set_title('t-SNE using labels from %s Clustering' % title1, fontsize = 16)

    ax[1].scatter(df[:,0], df[:,1], c = label2, cmap="RdBu")
    ax[1].set_title('t-SNE using labels from %s Clustering' % title2, fontsize = 16)

    fig.subplots_adjust(hspace=0.2)
    plt.show()


def plot_overview(df):
    # make a figure twice as wide as it is tall
    plt.figure(figsize=plt.figaspect(0.5))
    
    # save the 3 different plots created as 'figs'
    figs = plt.plot(df)
    # iterate over 'figs' and create a label for each one from the column names of t1_t2_clean
    plt.legend(iter(figs), list(df))

    plt.show()
    
    
def plot_boxplot(df, label):
    # Create a figure instance
    fig = plt.figure(figsize=(14,7))

    # Create the subplots, sharing the y axis
    ax1 = plt.subplot(121)
    plt.boxplot(df[label == 0].values)
    plt.setp(ax1, xticklabels=list(df))
    plt.title('Class I', fontsize=18)

    ax2 = plt.subplot(122, sharey=ax1)
    plt.boxplot(df[label == 1].values)
    plt.setp(ax2, xticklabels=list(df))
    plt.title('Class II', fontsize=18)

    plt.show()


def print_tab(df, label):
    # Use the labels of KMeans, as they gave a better separation.
    mean1 = df[label == 0].mean(axis=0)
    std1 = df[label == 0].std(axis=0)

    mean2 = df[label == 1].mean(axis=0)
    std2 = df[label == 1].std(axis=0)

    # Print the values in a table
    print( tabulate([['Class I', "%i +/- %i"%(mean1[0],std1[0]), "%i +/- %i"%(mean1[1],std1[1]), "%i +/- %i"%(mean1[2],std1[2])], 
                     ['Class II', "%i +/- %i"%(mean2[0],std2[0]), "%i +/- %i"%(mean2[1],std2[1]), "%i +/- %i"%(mean2[2],std2[2])]], 
                    headers=list(df), tablefmt="fancy_grid") )