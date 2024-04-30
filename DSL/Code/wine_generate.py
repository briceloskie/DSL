import csv


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_wine_data(path):
    df=pd.read_csv(path, header=None)
    data = np.array(df)
    labels=data[:,0]
    labels=labels_processing(labels)
    data=data[:,1:]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data,labels

def labels_processing(raw_labels):
    new_lables=[]
    labels_dict={}
    count=0
    for i in raw_labels:
        if i not in labels_dict:
            labels_dict[i]=count
            count=count+1
    for j in raw_labels:
        new_lables.append(labels_dict[j])
    return new_lables


if __name__ == '__main__':
    data,labels=generate_wine_data(path="wine.data")