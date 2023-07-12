# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:43:16 2023

@author: narim
"""

"Given the DF and list of attributes, removes the ouyliers using the IQR method"
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

"Given the DF runs the corrolation analysis for numericla atrs and returs the pearson and spearman coeficents"
def pearson_spearman(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    for column in df:
       "Range normalizing attributes"
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1).round(2)
    spearman=df.corr(method='spearman', min_periods=1).round(2)
    return(pearson,spearman)


"given the DF, returns list of numeric attributes"
def numeric_atr_list(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    numeric_atr=list(df.columns)
    return(numeric_atr)

"given the df, filters the data to numeric attributes then normalises the attributes by range normalization into 0 to 1 range"
"needs to import these libraries"
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
def normalize(df):
    num_atr=numeric_atr_list(df)
    df_numeric=df[num_atr]
    sc=StandardScaler().fit(df_numeric)
    df_str=sc.transform(df_numeric)
    df_str=pd.DataFrame(data=df_str,columns=df_numeric.columns,index=df_numeric.index)
    return(df_str)




"given a DF and number of PCAs; already normalised, outlier removed,and filtered to the numeric attributes without the target atr, runs PCA dimentionality reduction algorythm\
    returns"
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def pca(df,n):
    pca = PCA(n_components=n)
    pca.fit(df)
    atr_list=df.columns
    PCs= ["PC-" + str(i) for i in range(1, n + 1)]
    eigenvectors = abs(pd.DataFrame(data=pca.components_.transpose(),index=atr_list,columns=PCs))
    #variance Ratio is the amount of variance explained by each principal component
    variance_ratios = pd.DataFrame(data=pca.explained_variance_ratio_,index=PCs,columns=["VariationRatio"])
    #singular values are square roots of the eigenvalues
    singular_values = pd.DataFrame(data=pca.singular_values_,columns=["SquareRoot_EigenValues"],index=PCs)
    #data in the new feature space
    transformed_data = pd.DataFrame(data=pca.transform(df),columns=PCs,index=df.index)
    cumulative_variance = np.cumsum(variance_ratios)
    plt.scatter(range(1, len(variance_ratios) + 1), cumulative_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative PCs Variance')
    plt.show()
    return(eigenvectors,variance_ratios,singular_values,transformed_data,transformed_data,variance_ratios,cumulative_variance)

"Given DF, and eigenvectors returns the most essential list of attributes from PCA results while having removed the coliniarity"
def best_atrs_wh_coliniarity(df,df_eigenvectors):
    pearson,spearman=pearson_spearman(df)
    atr_list=[]
    for PC in df_eigenvectors.columns:
        sorted_eigenvectors=df_eigenvectors.sort_values(PC,ascending=False).head(5)
        for atr in sorted_eigenvectors.index.to_list():
            if ((atr not in atr_list) & (df_eigenvectors.loc[atr,PC]>0.3)):
                atr_list.append(atr)
    i,j=spearman.shape
    for row in range(1,i+1):
        spearman.iloc[row-1,0:row]=0
    spearman=abs(spearman)

    row_indexes = np.where(spearman>0.8)[0]
    column_indexes = np.where(spearman>0.8)[1]
    columns=spearman.columns
    corrolated_columns=columns[column_indexes]
    atr_list=[x for x in atr_list if x not in corrolated_columns]
    return (atr_list)
