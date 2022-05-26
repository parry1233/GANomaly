from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib

class SVM_classifier():
    def __init__(self,normal, abnormal, testSize):
        self.x_concat = np.concatenate((normal, abnormal), axis=None)
        print(self.x_concat)
        
        normal_len, abnormal_len = len(normal), len(abnormal)
        y_normal, y_abnormal = np.ones((normal_len,), dtype=int), np.zeros((abnormal_len,), dtype=int)
        self.y_concat = np.concatenate((y_normal, y_abnormal), axis=None)
        print(self.y_concat)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_concat, self.y_concat, test_size=testSize, random_state=0)
        #print(self.y_train)
        #print(self.y_test)
        print('Train data count = '+str(len(self.y_train)), 'Test data count = '+str(len(self.y_test)))
        
        self.svm = SVC(kernel='linear', probability=True)
        
    def train(self):
        X_train_reshape = self.X_train.reshape(-1,1)
        self.svm.fit(X_train_reshape, self.y_train)
    
    def load_model(self):
        self.svm = joblib.load('saved_model/SVM_model')
    
    def accuracy_on_test(self):
        X_test_reshape = self.X_test.reshape(-1,1)
        error = 0
        predict_result = self.svm.predict(X_test_reshape)
        for i, v in enumerate(predict_result):
            if v!= self.y_test[i]:
                error += 1
        print(predict_result)
        print('error = '+str(error)+' on '+str(len(self.y_test))+' test samples')
        print('accuracy = '+str(1 - (error / len(self.y_test))))
        
        joblib.dump(self.svm, 'saved_model/SVM_model')
        
        return self.X_test, predict_result, self.y_test