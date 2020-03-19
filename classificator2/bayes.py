# ======================================================
# Copyright (C) 2020 repa1030
# This program and the accompanying materials
# are made available under the terms of the MIT license.
# ======================================================
from math import sqrt
from math import pi
from math import exp
import operator

class BayesGaussian():

    # init
    def __init__(self, data_train, data_test, detection_classes):
        self.train_data = data_train
        self.test_data = data_test
        self.det_cl = detection_classes

    # calc mean of a list of numbers
    def mean(self, numbers):
        mean = sum(numbers) / float(len(numbers))
        return mean

    # calc sigma of a list of numbers
    # sigma = sqrt( sum from i to N [ (x_i - x_mean)^2 ] / (N-1) )
    def standard_devation(self, numbers):
        avg = self.mean(numbers)
        sum_num = 0.0
        for x in numbers:
            sum_num = sum_num + ((x - avg) * (x - avg))
        sig = sqrt(sum_num / float(len(numbers)-1))
        return sig

    # calc mean, sigma and count in dataset
    # this function returns a list where mean, sigma, 
    # and length of each class is represented
    def sum_dataset(self, dataset):
        summaries = [(self.mean(column), self.standard_devation(column), len(column)) for column in zip(*dataset)]
        # delete statistic for class variable
        del(summaries[0])
        return summaries

    # split data in class and calc statistics
    # input is a unsorted dataset which will be sorted
    # in the first step in a dictionary (keys are the classes)
    # then the statistics for each class (mean, sigma and row count)
    # are calculated and the results are stored in a new dictionary "summaries"
    def sum_class(self, dataset):
        separated_dict = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[0]
            if (class_value not in separated_dict):
                separated_dict[class_value] = list()
            separated_dict[class_value].append(vector)
        summaries = dict()
        for class_value, rows in separated_dict.items():
            summaries[class_value] = self.sum_dataset(rows)
        return summaries

    # calc Gaussian probability distribution function for x
    # f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
    def calc_probability(self, x, mean, sig):
        gauss = (1 / (sqrt(2 * pi) * sig)) * exp(-((x-mean)**2 / (2 * sig**2 )))
        return gauss

    # calc probabilities of predicting each class for a given row (data)
    def calc_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        # P(class|data)
        for class_value, class_summaries in summaries.items():
            # P(class)
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            i = 0
            # P(data|class) = P(data1|class) * P(data2|class) * ...
            while i < len(class_summaries):
                mean, sig, count = class_summaries[i]
                probabilities[class_value] *= self.calc_probability(row[i+1], mean, sig)
                i += 1
        return probabilities

    # prediction pipeline
    def predictClass(self):
        cnt_det = 0
        cnt_all = len(self.test_data)
        sums = self.sum_class(self.train_data)
        for test in self.test_data:
            # get probabilities
            probabilities = self.calc_class_probabilities(sums, test)
            # search highest probability in dictionary
            highest_prob = 0
            for key in probabilities:
                if (probabilities[key] > highest_prob):
                    highest_prob = probabilities[key]
                    pred = int(key)
            # display output
            dsp = 'Object class prediction: ' + str(int(pred)) + '\n'
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
        print('\nSummary Bayes: ' + str(cnt_det) + "/" + str(cnt_all) + ' objects are correctly classified.')
