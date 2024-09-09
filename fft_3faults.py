#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:35:11 2023

@author: jhao2
"""


import matplotlib.pyplot as plt
from comtrade import Comtrade
import numpy as np
from functions import listfiles
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
import scikitplot as skplt
import scipy.fft as FFT

fautl_types = ['f1','f2','f3']
labels = []
data = []
channels = [3,4,5]
N = 60
T=0.00005
for j in range(len(fautl_types)):
    # list all the files
    all_files, partial_files = listfiles('Faults/COMTRADE file', fautl_types[j])
    # fault happens at 0.3s and never cleared
    label = np.concatenate((np.zeros((1,6000)),np.ones((1,6001))*int(fautl_types[j][-1])), axis=1)[0]
    
    for i in range(len(partial_files)):
        
        file_name = partial_files[i][-44:-4]
        rec = Comtrade()
        rec.load(f"{file_name}.cfg", f"{file_name}.dat")
        print("Trigger time = {}s".format(rec.trigger_time))
        print(f'The label at time (index 6000) {rec.time[6000]} is {label[6000]}, the label at time (index 5999) {rec.time[5999]}  is {label[5999]}')
        
        tem_label = []
        tem_data = []
        tem_data_x = []
        tem_data_shift = []
        tem_data_abs = []
        # need channel 3,4,5
        for k in channels:
            t = 0
            tem_label_channel = []
            tem_data_channel = []
            tem_data_x_channel = []
            tem_data_shift_channel = []
            tem_data_abs_channel = []
            while t<12000:
                I = np.array(rec.analog[k][t:t+60])
                I_fft = FFT.fft(I)
                tem_data_channel.append(I_fft)
                tem_data_x_channel.append(FFT.fftfreq(N, T))
                tem_data_shift_channel.append(FFT.fftshift(I_fft))
                tem_data_abs_channel.append([np.real(tem_data_shift_channel[-1][30]),np.real(tem_data_shift_channel[-1][31])])
                if np.mean(label[t:t+60])>0.5:
                    tem_label_channel.append(int(fautl_types[j][-1]))
                else:
                    tem_label_channel.append(0)
                t+=60
            
            tem_data.append(np.array([np.array(xi) for xi in tem_data_channel]))
            tem_data_x.append(np.array([np.array(xi) for xi in tem_data_x_channel]))
            tem_data_shift.append(np.array([np.array(xi) for xi in tem_data_shift_channel]))
            tem_data_abs.append(np.array([np.array(xi) for xi in tem_data_abs_channel]))
            tem_label.append(np.array([np.array(xi) for xi in tem_label_channel]).reshape(-1,1))
    

        labels.append(tem_label[0])
        data.append(np.concatenate(tem_data_abs, axis=1))
    
# convert all data into (#sample, #feature, label)

X_data = np.concatenate(data)
y_data = np.concatenate(labels)

# Create a boolean mask indicating which rows have NaN values
nan_mask = np.isnan(X_data).any(axis=1)
# Create a new array containing only the rows without NaN values
X_data = X_data[~nan_mask]
y_data = y_data[~nan_mask]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=42)

# train the model
clf = RFC(n_estimators=100, max_depth=30, random_state=0)
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
print(f'The accuracy of the model is {accuracy_score(y_test.ravel(), y_pred)}')

# plot figures
plt.rcParams['figure.dpi'] = 900
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, title = 'Confusion Matrix for Fault Detection')

skplt.metrics.plot_roc(y_test, y_prob, title = 'ROC Plot for Fault Detection')

    
    


