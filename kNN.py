import numpy as np
import pandas as pd
from scipy.spatial import distance
train = pd.read_csv('./diabetes_data/實驗A/train_data.csv')
test =  pd.read_csv('./diabetes_data/實驗A/test_data.csv')

train_data = train[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
train_target = train[["Outcome"]]
test_data = test[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
test_target = test[["Outcome"]]

class kNN:
    def __init__(self,k):
        self.k = k

    def getData(self,train_data,train_target):
        self.train_data = train_data
        self.train_target = train_target

    def classify(self,test_data):
        result = []
        for i in range(len(test_data)):
            dist_arr = []
            tar_arr = []
            for j in range(len(train_data)):
                dist = distance.euclidean(train_data.iloc[j],test_data.iloc[i])
                dist_arr.append([dist,j])
            dist_arr.sort()
            dist_arr = dist_arr[0:self.k]#抓出最近的k個點
            # print(dist_arr)
            for a,b in dist_arr:#抓出k個training data 的 target
                tar_arr.append(train_target.iloc[b][0])
            cla_result = max(tar_arr,key = tar_arr.count)
            result.append(cla_result)
        #print(result)
        return result
    def accuracy(self,test_data,test_target):
        classify = self.classify(test_data)
        acc_arr = []
        for n in range(len(test_target)):
            acc = (classify[n]==test_target.iloc[n][0])
            acc_arr.append(acc)
        s = sum(acc_arr)/len(test_target)
        print(s)

exe = kNN(5)
exe.getData(train_data,train_target)
#exe.classify(test_data)
exe.accuracy(test_data,test_target)
