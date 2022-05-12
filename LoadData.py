from turtle import width
from keras.datasets import mnist
import cv2
import numpy as np

class LoadData():
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def reshape_x(self, x, width, height):
        new_x = np.empty((len(x), width, height))
        for i, e in enumerate(x):
            new_x[i] = cv2.resize(e, (width, height))
        return np.expand_dims(new_x, axis=-1) / 127 - 1
    
    def Define_normal_Abnormal(self, normal_class, abnormal_class_1, abnormal_class_2):
        self.x_ok = self.x_train[self.y_train == normal_class]
        self.x_test = self.x_test[(self.y_test==normal_class) | (self.y_test == abnormal_class_1) | (self.y_test == abnormal_class_2)]
        self.y_test = self.y_test[(self.y_test==normal_class) | (self.y_test == abnormal_class_1) | (self.y_test == abnormal_class_2)]
        
        print(self.y_test)
        
        return (self.x_ok, self.x_test, self.y_test)
    
    def getOriginalData(self):
        return (self.x_train, self.y_train)

