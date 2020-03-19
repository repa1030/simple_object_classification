# ======================================================
# Copyright (C) 2020 repa1030
# This program and the accompanying materials
# are made available under the terms of the MIT license.
# ======================================================

# import classificators
from scripts import generate_data as gtd
from classificator1 import k_nearest_neighbors as knn
from classificator2 import bayes
import numpy as np
import sys

# set false if you already have txt files with data for testing and training
# notice: generation of data may take more than 10 minutes
generate_train_data = True
generate_test_data = True

# define object classes
# format: ["string_in_file_name", class_id]
obj_classes = [
    ["santa1", 1],
    ["santa2", 2],
    ["snowman", 3],
    ["angel1", 4],
    ["angel2", 5],
    ["angel3", 6],
    ["bear1", 7],
    ["bear2", 8],
    ["bear3", 9],
    ["fail", -1]
]

# classes that have to be detected
# format: ["name", object_class_1, object_class_2, ...]
detection_classes = [
    ["santa", 1, 2],
    ["snowman", 3],
    ["angel", 4, 5, 6],
    ["bear", 7, 8, 9],
    ["fail", -1]
]

# good parameters:
# canny threshold c_th1: 100
# canny threshold c_th2: 200
# r: 0.05 (use r=1.0, if images are already small)
# color_th: 90
# num_train_data_to_use specifies the number of training data per object
# that is used to generate the data set. If all pictures should be used set it to -1

# training and test data set generation
dataGen = gtd.DataGenerator(obj_classes, train_pics="images/training",
                            test_pics="images/testing", r=1.0, c_th1=100,
                            c_th2=200, color_th=90, is_background_dark=True,
                            num_train_data_to_use=-1)

if generate_train_data:
    data_train = dataGen.generateTrainData()
else:
    data_train = np.loadtxt("training_data.txt", delimiter=',')

if generate_test_data:
    data_test = dataGen.generateTestData()
else:
    data_test = np.loadtxt("test_data.txt", delimiter=',')

# classificator K Nearest Neighbors

print('\n')
print('#####################################################')
print('################ K Nearest Neighbors ################')
print('#####################################################')
print('\n')

kNearNeigh = knn.KNearestNeighbor(data_train, data_test, detection_classes, k_neighbors=5)
kNearNeigh.predictClass()

# classificator Bayes

print('\n')
print('#####################################################')
print('####################### Bayes #######################')
print('#####################################################')
print('\n')

bay = bayes.BayesGaussian(data_train, data_test, detection_classes)
bay.predictClass()
