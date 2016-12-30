#!/usr/bin/env python
# -*- coding: utf-8 -*-
#This script generate rotation, translation, scaling and affined version of an image.
import cv2
from PIL import Image
from itertools import product
import os
import numpy as np
import random

DATA_ROOT = '/home/ywy/zy/data/train'
DATA_AUG_POS_SHIFT_MIN = -2
DATA_AUG_POS_SHIFT_MAX = 2
#DATA_AUG_SCALES = [0.9, 1.1]
DATA_AUG_ROT_MIN = -180
DATA_AUG_ROT_MAX = 180
divs = range(10,50,15)

def aug_pos(im, name, prefix):
    rect = {'cx':im.size[0]/2,'cy':im.size[1]/2,'wid':im.size[0]*4/5, 'hgt':im.size[1]*4/5}
    for sx, sy in product(
            range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX),
            range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX)):
        cx = rect['cx'] + sx
        cy = rect['cy'] + sy
        cropped_im = im.crop((cx - rect['wid'] // 2, cy - rect['hgt'] // 2,
                              cx + rect['wid'] // 2, cy + rect['hgt'] // 2))
        aug_pos_suffix = 'p' + str(sx) + str(sy)
        cropped_im.save(os.path.join(prefix, '_'.join([name.split('.')[0],aug_pos_suffix])+'.jpg'))

        pass
    pass

def aug_scale(im, name, prefix):
    rect = {'cx':im.size[0]/2,'cy':im.size[1]/2,'wid':im.size[0]*4/5, 'hgt':im.size[1]*4/5}
    for s in DATA_AUG_SCALES:
        w = int(rect['wid'] * s)
        h = int(rect['hgt'] * s)
        cropped_im = im.crop((rect['cx'] - w // 2, rect['cy'] - h // 2,
                              rect['cx'] + w // 2, rect['cy'] + h // 2))
        aug_scale_suffix = 's' + str(s)
        cropped_im.save(os.path.join(prefix, '_'.join([name.split('.')[0], aug_scale_suffix])+'.jpg'))
    pass

def aug_rot(im, name, prefix):
    rect = {'cx':im.size[0]/2,'cy':im.size[1]/2,'wid':im.size[0]*4/5, 'hgt':im.size[1]*4/5}
    for r in range(DATA_AUG_ROT_MIN, DATA_AUG_ROT_MAX,30):
        rotated_im = im.rotate(r)
        cropped_im = rotated_im.crop(
            (rect['cx'] - rect['wid'] // 2, rect['cy'] - rect['hgt'] // 2,
             rect['cx'] + rect['wid'] // 2, rect['cy'] + rect['hgt'] // 2))
        aug_rot_suffix = 'r' + str(r)
        cropped_im.save(os.path.join(prefix, '_'.join([name.split('.')[0], aug_rot_suffix])+'.jpg'))

    pass

def aug_affine(img, name, prefix):
    for div in divs:
        biasv = img.shape[1]//div
        biasu = img.shape[0]//div
        SrcPoints = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]]])
        CanvasPoints = np.float32([[0, 0], [img.shape[1] - biasv, biasv], [biasu, img.shape[0] - biasu]])
        SrcPointsA = np.array(SrcPoints, dtype=np.float32)
        CanvasPointsA = np.array(CanvasPoints, dtype=np.float32)
        AffinedImg = cv2.warpAffine(img, cv2.getAffineTransform(np.array(SrcPointsA),
                                                                np.array(CanvasPointsA)), (img.shape[1], img.shape[0]))
        AffinedImg=remove_black(AffinedImg)
        aug_affine_suffix = 'div_leading'+str(div)
        cv2.imwrite(os.path.join(prefix, '_'.join([name.split('.')[0], aug_affine_suffix])+'.jpg'),AffinedImg)

        SrcPoints = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]]])
        CanvasPoints = np.float32([[biasv, biasu], [img.shape[1], 0], [img.shape[1]-biasv, img.shape[0]-biasu]])
        SrcPointsB = np.array(SrcPoints, dtype=np.float32)
        CanvasPointsB = np.array(CanvasPoints, dtype=np.float32)
        AffinedImg = cv2.warpAffine(img, cv2.getAffineTransform(np.array(SrcPointsB),
                                                                np.array(CanvasPointsB)), (img.shape[1], img.shape[0]))
        AffinedImg = remove_black(AffinedImg)
        aug_affine_suffix = 'div_counter' + str(div)
        cv2.imwrite(os.path.join(prefix, '_'.join([name.split('.')[0], aug_affine_suffix]) + '.jpg'), AffinedImg)
    pass

def aug_gaussianblur(img, name, prefix):
    for sigma in [1.5,4.5]:
        kernel_size = (5, 5)

        GBimg = cv2.GaussianBlur(img, kernel_size, sigma)
        aug_GB_suffix = 'GaussianBlur_' + str(kernel_size[0])+ '_'+ str(sigma)
        cv2.imwrite(os.path.join(prefix, '_'.join([name.split('.')[0], aug_GB_suffix]) + '.jpg'), GBimg)


def remove_black (img):
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[i][j].all==0:
                img[i][j][0]=random.uniform(0,255)
                img[i][j][1] = random.uniform(0, 255)
                img[i][j][2] = random.uniform(0, 255)
    return img
# def data_aug(data_root):
#     # for dir in os.listdir(data_root):
#     pref = data_root
#     for item in os.listdir(data_root):
#         # print os.path.join(pref,item)
#         im = Image.open(os.path.join(pref,item))
#         # print im,im.size[0],im.size[1]
#         img = cv2.imread(os.path.join(pref, item))
#         fullname = item.split('.')[0]
#         name = fullname.split('_')[0]
#         aug_pos(im,name, pref)
#         aug_rot(im,name, pref)
#         aug_scale(im,name, pref)
#         aug_affine(img, name, pref)
#         aug_gaussianblur(img,name, pref)

if __name__ == '__main__':

    for item in os.listdir(DATA_ROOT):
        if item[:item.find('IMG')]=='0':
            im = Image.open(os.path.join(DATA_ROOT,item))
            img = cv2.imread(os.path.join(DATA_ROOT, item))
            #fullname = item.split('.')[0]
            #name = fullname.split('_')[0]
            #aug_pos(im,item, DATA_ROOT)
            aug_rot(im,item, DATA_ROOT)
    for item in os.listdir(DATA_ROOT):
        if item[:item.find('IMG')] == '0':
            im = Image.open(os.path.join(DATA_ROOT, item))
            img = cv2.imread(os.path.join(DATA_ROOT, item))
            #aug_scale(im,item, DATA_ROOT)
            aug_affine(img, item, DATA_ROOT)
            aug_gaussianblur(img,item, DATA_ROOT)
        pass
pass