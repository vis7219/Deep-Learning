# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:16:14 2021

@author: HP
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
from PIL import Image


import os
os.chdir('C:\\Users\\HP\\GitHub\\Deep-Learning')

np.seterr(all = 'ignore')

def load_dataset():
    with h5py.File('train_catvnoncat.h5' , 'r') as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    with h5py.File('test_catvnoncat.h5' , 'r') as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])
        
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return(train_set_x_orig , train_set_y_orig , test_set_x_orig , test_set_y_orig , classes)

# Initiating the dataset

train_X, train_Y, test_X, test_Y, classes = load_dataset()

train_X = train_X.reshape(train_X.shape[0] , -1).T
test_X = test_X.reshape(test_X.shape[0] , -1).T

#train_X = train_X/255
#test_X = test_X/255

# Creating helper function - Sigmoid Function, Output = 's'
def sigmoid(z):
    z_exp = np.exp(-z)
    s = 1/(1+z_exp)
    return(s)

# Initiating 'w' & 'b' as zeros; Output = 'w' & 'b'
def initialize_with_zeros(dim):
    w = np.zeros( (dim , 1), dtype = np.float64 )
    b = 0.0
    return (w,b)

# Function for 'Forward' and 'Backward' Propogation; Output = grads(dw , db) & cost
def propagate(w , b , X , Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T , X) + b)
    cost = np.sum(-(Y*np.log(A) + (1-Y)*np.log(1-A)))/m
    
    dw = (np.dot(X , (A-Y).T))/m
    db = np.sum(A-Y)/m
    
    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# Function for updating 'w' & 'b' by minimizing cost function; Output = params(w,b),grads(dw,db),costs
def optimize(w , b , X , y , num_iterations , learning_rate , print_cost):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    
    for i in range(num_iterations):
        grads , cost = propagate(w , b , X , y)
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# Function for predicting 'Yhat'; Output = y_predict
def predict(w , b , x):
    m = x.shape[1]
    Y_predict = np.zeros( (1 , m) )
    w = w.reshape(x.shape[0] , 1)
    
    A = sigmoid(np.dot(w.T , x) + b)
    Y_predict = np.around(A)
    return Y_predict

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    
    m = X_train.shape[0]
    w , b = initialize_with_zeros(m)
    
    params, grads, costs = optimize(w, b, X_train , Y_train, num_iterations, learning_rate, print_cost)
    w = params['w']
    b = params['b']
    
    Y_prediction_test = predict(w , b , X_test)
    Y_prediction_train = predict(w , b , X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


logistic_regression_model = model(train_X, train_Y, test_X, test_Y, 2000, 0.005, print_cost=True)


my_image = "13.jpg"   

#We preprocess the image to fit your algorithm.
fname = "C:\\Users\\HP\\GitHub\\Deep-Learning\\" + my_image
num_px = 64
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")








