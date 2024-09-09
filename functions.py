# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:22:05 2021

@author: jhao2
"""
import os, glob
import numpy as np
import pandas as pd
# from kneed import KneeLocator
# from sklearn.cluster import KMeans
# from joblib import dump, load

def listfiles(path, case=None):
    """
    Parameters
    path : folder in the current directory contains csv files.
    -------
    Returns
    files : file path for future pd.csv_read
    houses : House_id
    """
    files = []
    sub_files = []
    filepath = os.getcwd()+"/"+path
    files = glob.glob(filepath+"/*.cfg") # get all the csv files' name
    if not files:
        files = glob.glob(filepath+"/*.dat")
    if case:
        for i in files:
            if case in i:
                sub_files.append(i)
                
    return files, sub_files
def readncal(files,stats=['min', 'max', 'std'],cols=['Cooling Setpoint Controlled Output_1()','Heating Setpoint Controlled Output_1()']):
    """
    Parameters
    ----------
    files : list of string
        csv file names -- data.
    stats : math, optional
        DESCRIPTION. The default is ['min', 'max', 'std'], evaluate data.
    cols : column name, optional
        DESCRIPTION. The default is ['grid_l1','grid_l2'], what columns of data to be processed.

    Returns
    -------
    X : Matrix of stats of data, each row corresponds to one file in files
        DESCRIPTION.

    """
    X = np.zeros((len(files),len(stats)*len(cols)))
    for i in range(len(files)):
        #print(i)
        profile = pd.read_csv(files[i],usecols=cols, dtype='float64')
        m = 0
        for j in cols:
            for k in stats:
                expr = 'profile['+'\''+j+'\''+'].'+k+'()'
                X[i][m]=(eval(expr))
                m+=1
        
    return X

# def srchcluster(X,n_cluster,kmeans_kwargs):
#     """
#     Parameters
#     ----------
#     X : np.array
#         data for KMeans.
#     n_cluster : int
#         can't be larger than len of X.
#     kmeans_kwargs: dictionary
#         save the hyperparameters for kmeans
#     Returns
#     -------
#     sse : list
#         Sum of squared error for different number of clusters.
#     """

#     sse = []
#     for k in range(1, n_cluster):
#         kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#         kmeans.fit(X)
#         sse.append(kmeans.inertia_)
#     return sse
        
# def k_means(X,o_cluster,kmeans_kwargs):
#     kmeans_model = KMeans(n_clusters=o_cluster, **kmeans_kwargs)
#     kmeans_model.fit(X)
#     return kmeans_model

# def homeclusters(k_model,home_id):
#     """
#     This function returns the groups by kmeans models and houses in each cluster
#     ----------
#     Parameters
#     ----------
#     k_model : KMeans model
#         Trained KMneans model.
#     home_id : list
#         list of home_id.
#     ----------
#     Returns
#     -------
#     clusters : dictionary
#         key:cluster, value: home id in the cluster.

#     """
#     clusters = {}
#     n_cluster = np.unique(k_model.labels_)
#     for i in n_cluster:
#         positions = np.where(k_model.labels_==i)
#         ids = list(np.array(home_id)[positions[0]])
#         for j in range(len(ids)):
#             ids[j] = ids[j].tolist()
#         clusters[i] = ids
#     return clusters

# def load_model(modelid,name=['air','car','solar']):
#     model = {}
#     for i in name:
#         # file = i+'.joblib'
#         model[i] = load('model/'+modelid+'/'+i+'.joblib')
#     return model



