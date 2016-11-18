#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from itertools import product
import os

DATA_ROOT = '../Data/rcdata'
DATA_AUG_POS_SHIFT_MIN = -2
DATA_AUG_POS_SHIFT_MAX = 2
DATA_AUG_SCALES = [0.9, 1.1]
DATA_AUG_ROT_MIN = -15
DATA_AUG_ROT_MAX = 15

def aug_pos(im, name, prefix):
    aug_pos_ims = []
    aug_pos_suffixes = []

    rect = {'cx':im.size[0]/2,'cy':im.size[1]/2,'wid':im.size[0]*4/5, 'hgt':im.size[1]*4/5}
    for sx, sy in product(
            range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX),
            range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX)):
        cx = rect['cx'] + sx
        cy = rect['cy'] + sy
        cropped_im = im.crop((cx - rect['wid'] // 2, cy - rect['hgt'] // 2,
                              cx + rect['wid'] // 2, cy + rect['hgt'] // 2))
        # aug_pos_ims.append(cropped_im)
        # aug_pos_suffixes.append('p' + str(sx) + str(sy))
        aug_pos_suffix = 'p' + str(sx) + str(sy)
        # print os.path.join(prefix, '_'.join([name,aug_pos_suffix])+'.jpg')
        cropped_im.save(os.path.join(prefix, '_'.join([name,aug_pos_suffix])+'.jpg'))
        # cropped_im.save('1.jpg')
        pass
    pass
    # return aug_pos_ims, aug_pos_suffixes
    # return cropped_im, '_'.join(name,aug_pos_suffix)

def aug_scale(im, name, prefix):
    aug_scale_ims = []
    aug_scale_suffixes = []

    rect = {'cx':im.size[0]/2,'cy':im.size[1]/2,'wid':im.size[0]*4/5, 'hgt':im.size[1]*4/5}
    for s in DATA_AUG_SCALES:
        w = int(rect['wid'] * s)
        h = int(rect['hgt'] * s)
        cropped_im = im.crop((rect['cx'] - w // 2, rect['cy'] - h // 2,
                              rect['cx'] + w // 2, rect['cy'] + h // 2))
        # aug_scale_ims.append(cropped_im)
        # aug_scale_suffixes.append('s' + str(s))
        aug_scale_suffix = 's' + str(s)
        cropped_im.save(os.path.join(prefix, '_'.join([name, aug_scale_suffix])+'.jpg'))
    pass
    # return cropped_im, '_'.join(name,aug_scale_suffix)

def aug_rot(im, name, prefix):
    aug_rot_ims = []
    aug_rot_suffixes = []
    rect = {'cx':im.size[0]/2,'cy':im.size[1]/2,'wid':im.size[0]*4/5, 'hgt':im.size[1]*4/5}
    for r in range(DATA_AUG_ROT_MIN, DATA_AUG_ROT_MAX):
        rotated_im = im.rotate(r)
        cropped_im = rotated_im.crop(
            (rect['cx'] - rect['wid'] // 2, rect['cy'] - rect['hgt'] // 2,
             rect['cx'] + rect['wid'] // 2, rect['cy'] + rect['hgt'] // 2))
        # aug_rot_ims.append(cropped_im)
        # aug_rot_suffixes.append('r' + str(r))
        aug_rot_suffix = 'r' + str(r)
        cropped_im.save(os.path.join(prefix, '_'.join([name, aug_rot_suffix])+'.jpg'))
    pass
    # return cropped_im, '_'.join(name,aug_rot_suffix)

if __name__ == '__main__':
    for dir in os.listdir(DATA_ROOT):
        pref = os.path.join(DATA_ROOT, dir)
        for item in os.listdir(os.path.join(DATA_ROOT,dir)):
            im = Image.open(os.path.join(pref,item))
            name = item.split('_')[0]
            aug_pos(im,name, pref)
            aug_rot(im,name, pref)
            aug_scale(im,name, pref)
            pass
    pass