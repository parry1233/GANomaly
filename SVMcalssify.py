from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
        
    def accuracy_on_test(self):
        X_test_reshape = self.X_test.reshape(-1,1)
        error = 0
        predict_result = self.svm.predict(X_test_reshape)
        for i, v in enumerate(predict_result):
            if v!= self.y_test[i]:
                error += 1
        print(predict_result)
        print(error, len(self.y_test))
        print(1 - (error / len(self.y_test)))
        
        return self.X_test, predict_result, self.y_test
    
    '''
    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        alpha=0.6, 
                        c=cmap(idx),
                        edgecolor='black',
                        marker=markers[idx], 
                        label=cl)

        # highlight test samples
        if test_idx:
            # plot all samples
                X_test, y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        alpha=1.0,
                        edgecolor='black',
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')
    '''
    
    
        
        
    