import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
dataset = input("please enter the dataset: A ? B\n")
# number = input("enter k:")

train = pd.read_csv('./diabetes_data/實驗'+dataset+'/train_data.csv')
test =  pd.read_csv('./diabetes_data/實驗'+dataset+'/test_data.csv')

train_data = train[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
train_target = train[["Outcome"]]
test_data = test[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
test_target = test[["Outcome"]]

class kNN:
    def __init__(self,k):
        self.k = k

    def inverse(self,dist):
        return 1/dist

    def getData(self,train_data,train_target):
        self.train_data = train_data
        self.train_target = train_target

    def classify(self,test_data):
        result = []
        for i in range(len(test_data)):
            dist_arr = []
            tar_arr = [] #test data 旁 n 個點的分類
            for j in range(len(train_data)):
                dist = distance.euclidean(train_data.iloc[j],test_data.iloc[i])
                dist_arr.append([dist,j])
            dist_arr.sort()
            dist_arr = dist_arr[0:self.k]#抓出最近的k個點
            for a,b in dist_arr:#抓出k個training data 的 target            
                weight = self.inverse(a)
                tar_arr.append([weight,train_target.iloc[b][0]])             
            #投票    
            total_no = int(0)
            total_yes = int(0)
            for n,m in tar_arr:
                if m == 0:
                    total_no += n
                else:
                    total_yes += n
            cla_result = 0 if total_no > total_yes else 1
            result.append(cla_result)
        #print(result)
        return result
    def classifyNoWeight(self,test_data):
        result = []
        for i in range(len(test_data)):
            dist_arr = []
            tar_arr = [] #test data 旁 n 個點的分類
            for j in range(len(train_data)):
                dist = distance.euclidean(train_data.iloc[j],test_data.iloc[i])
                dist_arr.append([dist,j])
            dist_arr.sort()
            dist_arr = dist_arr[0:self.k]#抓出最近的k個點
            for a,b in dist_arr:#抓出k個training data 的 target            
                tar_arr.append(train_target.iloc[b][0])             
            #投票    
            cla_result = max(tar_arr,key=tar_arr.count)
            result.append(cla_result)
        #print(result)
        return result
    def confusionMatrix(self,test_data,test_target):
        predict = self.classify(test_data)
        actual = []
        for n in range(len(test_target)):
            temp = test_target.iloc[n][0]
            actual.append(temp)
        tn, fp, fn, tp = confusion_matrix(actual, predict).ravel()
        accuracy = (tp+tn)/(tp+fp+fn+tn)
        precision = tp/(tp+fp)#預測有糖尿病的情況下，正確的機率
        recall = tp/(tp+fn)#實際有糖尿病的情況下，預測真正有糖尿病的機率
        print("dataset %c" %(dataset))
        print("when k = %d:\naccuracy = %f\nprecision = %f\nrecall = %f"%(self.k,accuracy,precision,recall))
        return [accuracy,precision,recall]

    def accuracy(self,test_data,test_target):
        classify = self.classify(test_data)
        acc_arr = []
        for n in range(len(test_target)):
            acc = (classify[n]==test_target.iloc[n][0])
            acc_arr.append(acc)
        s = sum(acc_arr)/len(test_target)
        # print("dataset %c" %(dataset))
        # print("when k = %d ,model's accuracy: %f " %(self.k,s))
        return s
    def accuracy2(self,test_data,test_target):
        classify = self.classifyNoWeight(test_data)
        acc_arr = []
        for n in range(len(test_target)):
            acc = (classify[n]==test_target.iloc[n][0])
            acc_arr.append(acc)
        s = sum(acc_arr)/len(test_target)
        # print("dataset %c" %(dataset))
        # print("when k = %d ,model's accuracy: %f " %(self.k,s))
        return s
                
            
accu = []
accu2 = []
# prec = []
# reca = []
for number in range(1,100):
    exe = kNN(number)
    exe.getData(train_data,train_target)
    result1 = exe.accuracy(test_data,test_target)
    result2 = exe.accuracy2(test_data,test_target)
    accu.append(result1)
    accu2.append(result2)
# exe.confusionMatrix(test_data,test_target)
    # accu.append(result[0])
    # prec.append(result[1])
    # reca.append(result[2])
k_range = range(1,100)
# plt.plot(k_range, accu,label="Accuracy")
# plt.plot(k_range, prec,label="Precision")
# plt.plot(k_range, reca,label="Recall")
plt.plot(k_range,accu,color="red")
plt.plot(k_range,accu2)
plt.show()
# exe.accuracy(test_data,test_target)