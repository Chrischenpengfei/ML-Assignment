import numpy as np # linear algebra
import pandas as pd # data processing
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization

# Preprocessing, modelling and evaluating
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

import gc

# Output a summary table of a given dataframe. Get this good idea from other briliant kernels.
def summary(df):

    # give out the dimensional infomation of the data-frame
    print(f"Dimensionality of the DataFrame: {df.shape}") 

    # create a table based on data-type of each column of the input data-frame.
    table = pd.DataFrame(df.dtypes,columns=['dtypes'])

    # Reset the index of the DataFrame, and use the default index (column name) instead. Change the column name 'index' to 'name' and drop the index column.
    table = table.reset_index() 
    table['Name'] = table['index']
    table = table[['Name','dtypes']]

    # Add new column for the summary table.
    table['Missing'] = df.isnull().sum().values # check if any column in the input df has missing column.
    table['Uniques'] = df.nunique().values      # number of unique value in the column
    table['First Value'] = df.loc[0].values     # 1st value of the column
    table['Second Value'] = df.loc[1].values    # 2nd value of the column
    table['Third Value'] = df.loc[2].values     # 3rd value of the column

    # Study the randomness of each column, use scipy.stats.entropy, calculated by S = -sum(pk * log(pk), axis=axis)
    for name in table['Name'].value_counts().index:
        table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return table

''' Depending on your environment, pandas automatically creates int32, int64, float32 or float64 columns for numeric data. If you know the min or max value of a column, you can use a subtype which is less memory consuming. This function will check the min and max value of each int and float data from the dataset and assign an approperate smaller data type for saving the memory usage'''

def reduce_ram(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2 # original ram usage in MB
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            # get min and max of the column
            c_min = df[col].min()
            c_max = df[col].max()

            # comapre the value of each subtype with min and max of the col, to find the suitable subtype for shrinking the ram usage.
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2 # final ram usage in MB
    
    if verbose: print('Ram usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

total = len(df_train)
plt.figure(figsize=(12,6))

g = sns.countplot(x='target', data=df_train, color='green') #Show the counts of observations in each categorical bin using bars.
g.set_title("TARGET DISTRIBUTION", fontsize = 20)
g.set_xlabel("Target Vaues", fontsize = 15)
g.set_ylabel("Count", fontsize = 15)
sizes=[] # Get highest values in y
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

plt.show()

def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):
    
    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart
    plt.figure(figsize=(17, 35)) # size of figure

    # loop to get column and the count of plots
    for n, col in enumerate(df_train[cols]): 
        # Compute a simple cross tabulation of two (or more) factors. By default computes a frequency table of the factors unless an array of values and an aggregation function are passed. 
        tmp = pd.crosstab(df_train[col], df_train['target'], normalize='index') * 100
        tmp = tmp.reset_index()
        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

        ax = plt.subplot(grid[n]) # feeding the figure of grid
        sns.countplot(x=col, data=df_train, order=list(tmp[col].values) , color='green') 
        ax.set_ylabel('Count', fontsize=15) # y axis label
        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label
        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

        # twinX - to build a second yaxis
        gt = ax.twinx()
        gt = sns.pointplot(x=col, y='Yes', data=tmp,
                           order=list(tmp[col].values),
                           color='black', legend=False)
        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)
        gt.set_ylabel("Target %True(1)", fontsize=16)
        sizes=[] # Get highest values in y
        for p in ax.patches: # loop to all objects
            height = p.get_height()
            sizes.append(height)
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(height/total*100),
                    ha="center", fontsize=14) 
        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights


    plt.subplots_adjust(hspace = 0.5, wspace=.3)
    plt.show()