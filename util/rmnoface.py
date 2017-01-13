#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import time
import dlib

detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':
    start =time.time()
    data_path = '/system2/data/faces/chinesestar'
    for item in os.listdir(data_path):
        cls_name = os.path.join(data_path,item)
        for file in os.listdir(cls_name):
            file_name = os.path.join(cls_name,file)
            im = cv2.imread(file_name)
            rects = detector(im, 1)

            if len(rects) > 1:
                print('TooManyFaces')
                os.remove(file_name)
            if len(rects) == 0:
                print('NoFaces')
                os.remove(file_name)

    print ('spend ', time.time()-start, 'seconds!')