from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class SVM_classifier():
    def __init__(self,normal, abnormal, testSize):
        self.x_concat = np.concatenate((normal, abnormal), axis=None)
        #print(self.x_concat)
        
        normal_len, abnormal_len = len(normal), len(abnormal)
        y_normal, y_abnormal = np.ones((normal_len,), dtype=int), np.zeros((abnormal_len,), dtype=int)
        self.y_concat = np.concatenate((y_normal, y_abnormal), axis=None)
        #print(self.y_concat)
        
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
        #print(predict_result)
        print('error = '+str(error)+' on '+str(len(self.y_test))+' test samples')
        print('accuracy = '+str(1 - (error / len(self.y_test))))
        
        joblib.dump(self.svm, 'saved_model/SVM_model')
        
        return self.X_test, predict_result, self.y_test
    
    def ConfusionMartrix(self):
        X_train_reshape = self.X_train.reshape(-1,1)
        X_test_reshape = self.X_test.reshape(-1,1)
        train_predictions = self.svm.predict(X_train_reshape)
        test_predictions = self.svm.predict(X_test_reshape)
        train_confu_matrix = confusion_matrix(self.y_train, train_predictions)
        test_confu_matrix = confusion_matrix(self.y_test, test_predictions)
        
        fault_type = np.array(list(set(self.y_test)))
        print(fault_type)

        plt.figure(1,figsize=(18,8))

        plt.subplot(121)
        sns.heatmap(train_confu_matrix, annot= True,fmt = "d",
        xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
        plt.title('Training Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.subplot(122)

        plt.subplot(122)
        sns.heatmap(test_confu_matrix, annot = True,
        xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
        plt.title('Test Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.show()
        
    def ClassificationReport(self):
        X_test_reshape = self.X_test.reshape(-1,1)
        test_predictions = self.svm.predict(X_test_reshape)
        class_report = classification_report(y_pred=test_predictions, y_true=self.y_test)
        print(class_report)
        auc = metrics.roc_auc_score(y_true=self.y_test, y_score=test_predictions)
        print('AUC = '+str(auc))
        