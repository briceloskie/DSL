import numpy as np
from sklearn.datasets._kddcup99 import _fetch_brute_kddcup99


def generate_kddcup99_data():
    data = _fetch_brute_kddcup99(data_home=None, download_if_missing=True, percent10=False)
    X = data.data  # 特征数据
    X = np.delete(X, [1,2,3], axis=1)
    y = data.target  # 标签数据
    y = labels_processing(y)
    return X,y

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
    data,labels=generate_kddcup99_data()

