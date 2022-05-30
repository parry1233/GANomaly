from turtle import width
from keras.datasets import mnist
import cv2
import numpy as np
from numpy import squeeze
import pandas as pd
import sklearn.utils as skUtils

import scipy.io as sio

class LoadData():
    def __init__(self):
        self.dataset = np.load('dataset/CWRU_48k_load_1_CNN_data.npz')
        self.X = self.dataset['data']
        self.Y = self.dataset['labels']
        self.data_sort_by_category()
    
    def data_sort_by_category(self):
        self.category = set(self.Y)
        category_to_index = {
            'Normal': 1,
            'Ball_007': -1,
            'Ball_014': -2,
            'Ball_021': -3,
            'OR_007': -4,
            'OR_014': -5,
            'OR_021': -6,
            'IR_007': -7,
            'IR_014': -8,
            'IR_021': -9
        }
        self.x_dict, self.y_dict = {},{}
        for category in set(self.Y):
            #print(category)
            self.x_dict[category] = self.X[self.Y == category]
            #self.y_dict[category] = self.Y[self.Y == category]
            self.index_y = np.array(self.Y)
            self.index_y[self.index_y == category] = category_to_index[category]
            #print(self.index_y)
            self.y_dict[category] = self.index_y[self.Y == category].astype(int)
            #print(self.y_dict[category])
            #print(self.x_dict[category].shape, self.y_dict[category].shape)
        #print()
        #print(self.x_dict, self.y_dict)
        
    def train_test_split(self, rate = 0.8):
        x_normal_shuffle, y_normal_shuffle = skUtils.shuffle(self.x_dict['Normal'], self.y_dict['Normal'])
        self.x_train, self.x_normal_test = x_normal_shuffle[:int(len(x_normal_shuffle)*rate)], x_normal_shuffle[int(len(x_normal_shuffle)*rate):]
        self.y_train, self.y_normal_test = y_normal_shuffle[:int(len(y_normal_shuffle)*rate)], y_normal_shuffle[int(len(y_normal_shuffle)*rate):]
        #print(self.x_train.shape, self.x_normal_test.shape)
        #print(self.y_train.shape, self.y_normal_test.shape)
        
        abnormal_category = ['Ball_007', 'Ball_014', 'Ball_021', 'OR_007', 'OR_014', 'OR_021', 'IR_007', 'IR_014', 'IR_021']
        abnormal_x_test, abnormal_y_test = {}, {}
        each_abnormal_test_data = int(len(x_normal_shuffle)*(1-rate) / (len(self.y_dict.keys())-1) )
        for category in abnormal_category:
            abnormal_x_test[category] = self.x_dict[category][:each_abnormal_test_data]
            abnormal_y_test[category] = self.y_dict[category][:each_abnormal_test_data]
            
        
        xTest = np.concatenate([self.x_normal_test,
                                abnormal_x_test['Ball_007'],
                                abnormal_x_test['Ball_014'],
                                abnormal_x_test['Ball_021'],
                                abnormal_x_test['OR_007'],
                                abnormal_x_test['OR_014'],
                                abnormal_x_test['OR_021'],
                                abnormal_x_test['IR_007'],
                                abnormal_x_test['IR_014'],
                                abnormal_x_test['IR_021']])
        yTest = np.concatenate([self.y_normal_test,
                                abnormal_y_test['Ball_007'],
                                abnormal_y_test['Ball_014'],
                                abnormal_y_test['Ball_021'],
                                abnormal_y_test['OR_007'],
                                abnormal_y_test['OR_014'],
                                abnormal_y_test['OR_021'],
                                abnormal_y_test['IR_007'],
                                abnormal_y_test['IR_014'],
                                abnormal_y_test['IR_021']])
        self.x_test, self.y_test = skUtils.shuffle(xTest, yTest)
        #print(self.x_test.shape, self.y_test.shape)
        #print(self.y_test)
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def SVC_dataPrepare(self):
        abnormal_category = ['Ball_007', 'Ball_014', 'Ball_021', 'OR_007', 'OR_014', 'OR_021', 'IR_007', 'IR_014', 'IR_021']
        abnormal_x, abnormal_y = {}, {}
        each_abnormal_data = int(len(self.x_dict['Normal']) / (len(self.y_dict.keys())-1) )
        for category in abnormal_category:
            abnormal_x[category] = self.x_dict[category][:each_abnormal_data]
            abnormal_y[category] = self.y_dict[category][:each_abnormal_data]
        
        xdata = np.concatenate([self.x_dict['Normal'],
                                abnormal_x['Ball_007'],
                                abnormal_x['Ball_014'],
                                abnormal_x['Ball_021'],
                                abnormal_x['OR_007'],
                                abnormal_x['OR_014'],
                                abnormal_x['OR_021'],
                                abnormal_x['IR_007'],
                                abnormal_x['IR_014'],
                                abnormal_x['IR_021']])
        ydata = np.concatenate([self.y_dict['Normal'],
                                abnormal_y['Ball_007'],
                                abnormal_y['Ball_014'],
                                abnormal_y['Ball_021'],
                                abnormal_y['OR_007'],
                                abnormal_y['OR_014'],
                                abnormal_y['OR_021'],
                                abnormal_y['IR_007'],
                                abnormal_y['IR_014'],
                                abnormal_y['IR_021']])
        return (xdata, ydata)
        
    
    def Define_normal_Abnormal(self, normal_class, abnormal_class_1, abnormal_class_2):
        self.x_ok = self.x_train[self.y_train == normal_class]
        self.x_test = self.x_test[(self.y_test==normal_class) | (self.y_test == abnormal_class_1) | (self.y_test == abnormal_class_2)]
        self.y_test = self.y_test[(self.y_test==normal_class) | (self.y_test == abnormal_class_1) | (self.y_test == abnormal_class_2)]
        
        #print(self.y_test)
        
        return (self.x_ok, self.x_test, self.y_test)
    
    def getOriginalData(self):
        return (self.x_train, self.y_train)
    

import cwru_py3 as cwru

if __name__ == "__main__":
    loaddata = LoadData()
    loaddata.train_test_split()
    (x_train, y_train), (x_test, y_test) = loaddata.train_test_split()
    print(x_train.shape, y_train.shape)
    #print(x_train)
    #print(y_train)
    print(x_test.shape, y_test.shape)
    #print(y_test[y_test==-6])
    
    mat = sio.loadmat('dataset/Normal_2.mat',squeeze_me=True)
    print(mat.keys())
    
    
    '''
    data = cwru.CWRU(exp="12DriveEndFault", rpm="1797", length=384)
    print(data.nclasses)
    print(data.labels)
    arr_yTrain = np.array(data.y_train)
    print(np.array(arr_yTrain))
    print(arr_yTrain[arr_yTrain==15].shape)
    print(data.X_train[arr_yTrain==15].shape)
    '''