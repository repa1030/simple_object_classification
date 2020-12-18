# ======================================================
# Copyright (C) 2020 repa1030
# This program and the accompanying materials
# are made available under the terms of the MIT license.
# ======================================================
import numpy as np
import os
import math

class KNearestNeighbor:

    def __init__(self, train_data, test_data, detection_classes, k_neighbors=3):
        self.data_train = train_data
        self.data_test = test_data
        self.k = k_neighbors
        self.det_cl = detection_classes

    def predictClass(self):
        # foreach test data
        cnt_all = len(self.data_test)
        cnt_det = 0
        for test in self.data_test:
            # initialize
            dsts = []
            neighbors = []
            i = 0
            # create a list with the distance from test data to each train data
            for data in self.data_train:
                dst = math.sqrt((test[1] - data[1])**2
                                + (test[2] - data[2])**2
                                + (test[3] - data[3])**2
                                + (test[4] - data[4])**2
                                )
                dsts.append((data, dst))
            # sort the list with distances in ascending order
            dsts.sort(key=lambda tup: tup[1])
            # create a list that contains the k nearest neighbors to the test data
            while i < self.k:
                neighbors.append(dsts[i][0])
                i += 1
            # read the classes of the neighbors
            out = [row[0] for row in neighbors]
            # save the most common value in "pred"
            pred = max(set(out), key=out.count)
            pred_ct = out.count(pred)
            percent = int(float(pred_ct) / self.k * 100.0)
            # display output
            dsp = 'Nearest Neighbors: ' + str(out) + '\n'
            dsp = dsp + 'Object class prediction: ' + str(int(pred)) + ' (' + str(percent) + ' %)\n'
            dsp = dsp + 'Object class ground truth: ' + str(int(test[0])) + '\n'
            cl_set = False
            for cl in self.det_cl:
                if pred in cl:
                    pred_obj = cl[0]
                if test[0] in cl:
                    gt_obj = cl[0]
            if gt_obj == pred_obj:
                dsp = dsp + 'This object was correctly classified as ' + gt_obj + '\n'
                cnt_det += 1
            else:
                dsp = dsp + 'This object was falsely classified as ' + pred_obj + '\n'
            dsp = dsp + '####################'
            print(dsp)
        print('\nSummary K Nearest Neighbor: ' + str(cnt_det) + "/" + str(cnt_all) + ' objects are correctly classified.')
