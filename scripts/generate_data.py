# ======================================================
# Copyright (C) 2020 repa1030
# This program and the accompanying materials
# are made available under the terms of the MIT license.
# ======================================================
import cv2
import os
import numpy as np
from tqdm import tqdm

class DataGenerator:

    def __init__(self, obj_classes, train_pics="images/trainig",
                    test_pics="images/testing", r=0.05, c_th1=100, 
                    c_th2=200, color_th=90, is_background_dark=True,
                    num_train_data_to_use=-1):
        self.tr_folder = train_pics
        self.te_folder = test_pics
        self.resize = r
        self.cthresh1 = c_th1
        self.cthresh2 = c_th2
        self.color_thresh = color_th
        # Format: ["string_in_file_name", class_index]
        self.classes = obj_classes
        self.is_bg_dark = is_background_dark
        self.num_train_data = num_train_data_to_use
        
    def getCannyPic(self, img):
        img_re = cv2.resize(img, (int(img.shape[1] * self.resize), int(img.shape[0] * self.resize)))
        gray = cv2.cvtColor(img_re, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, self.cthresh1, self.cthresh2)
        return canny, gray, img_re

    def cleanWorkspace(self, folder):
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                os.remove(os.path.join(folder, file))
                continue
            f_str = str(file)       
            if (f_str.find("_canny") is not -1):
                os.remove(os.path.join(folder, file))
                continue
            if (f_str.find("_gray") is not -1):
                os.remove(os.path.join(folder, file))

    def initDictionary(self):
        dict_count = dict()
        for cl in self.classes:
            dict_count[cl[1]] = 0
        return dict_count

    def generateTrainData(self):
        self.cleanWorkspace(self.tr_folder)
        dict_count = self.initDictionary()
        data_str = ""
        train_data = []
        print('##### Start creating data set of training data...')
        k = 1
        for file in tqdm(os.listdir(self.tr_folder)):
            file_str = str(file)
            file_path = os.path.join(self.tr_folder, file)
            cl = -1
            for obj_cl in self.classes:
                if file_str.find(obj_cl[0]) is not -1:
                    cl = obj_cl[1]
                    dict_count[cl] += 1
                    break
            if dict_count[cl] > self.num_train_data and self.num_train_data is not -1:
                continue
            img = cv2.imread(file_path)
            img_canny, img_gray, img_re = self.getCannyPic(img)
            colorBuf = [0, 0, 0] #BGR
            count = 0
            for i in range(0, img_re.shape[0]):
                for j in range(0, img_re.shape[1]):
                    if self.is_bg_dark:
                        if (img_gray[i, j] > self.color_thresh):
                            colorBuf = [colorBuf[0] + img_re[i, j, 0], 
                                        colorBuf[1] + img_re[i, j, 1], 
                                        colorBuf[2] + img_re[i, j, 2]]                 
                            count += 1
                    else:
                        if (img_gray[i, j] < self.color_thresh):
                            colorBuf = [colorBuf[0] + img_re[i, j, 0], 
                                        colorBuf[1] + img_re[i, j, 1], 
                                        colorBuf[2] + img_re[i, j, 2]]               
                            count += 1
            color = [colorBuf[0]/count, colorBuf[1]/count, colorBuf[2]/count]
            nzCount = cv2.countNonZero(img_canny)
            data_str = data_str + str(cl) + "," + str(nzCount) + "," + str(color[0]) + "," + str(color[1]) + "," + str(color[2]) + "\n"
            train_data.append([cl, nzCount, color[0], color[1], color[2]])
        file_obj = open("training_data.txt","w+")
        file_obj.write(data_str)
        print('##### Data set of training data is done.')
        return train_data

    def generateTestData(self):
        test_data = []
        data_str = ''
        print('##### Start creating data set of test data...')
        for file in tqdm(os.listdir(self.te_folder)):
            file_str = str(file)
            file_path = os.path.join(self.te_folder, file)
            cl = -1
            for obj_cl in self.classes:
                if file_str.find(obj_cl[0]) is not -1:
                    cl = obj_cl[1]
                    break
            img = cv2.imread(file_path)
            img_canny, img_gray, img_re = self.getCannyPic(img)
            colorBuf = [0, 0, 0] #BGR
            count = 0
            for i in range(0, img_re.shape[0]):
                for j in range(0, img_re.shape[1]):
                    if self.is_bg_dark:
                        if (img_gray[i, j] > self.color_thresh):
                            colorBuf = [colorBuf[0] + img_re[i, j, 0], 
                                        colorBuf[1] + img_re[i, j, 1], 
                                        colorBuf[2] + img_re[i, j, 2]]                
                            count += 1
                    else:
                        if (img_gray[i, j] < self.color_thresh):
                            colorBuf = [colorBuf[0] + img_re[i, j, 0], 
                                        colorBuf[1] + img_re[i, j, 1], 
                                        colorBuf[2] + img_re[i, j, 2]]                  
                            count += 1
            color = [colorBuf[0]/count, colorBuf[1]/count, colorBuf[2]/count]
            nzCount = cv2.countNonZero(img_canny)
            data_str = data_str + str(cl) + "," + str(nzCount) + "," + str(color[0]) + "," + str(color[1]) + "," + str(color[2]) + "\n"
            test_data.append([cl, nzCount, color[0], color[1], color[2]])
        file_obj = open("test_data.txt","w+")
        file_obj.write(data_str)
        print('##### Data set of test data is done.')
        return test_data

