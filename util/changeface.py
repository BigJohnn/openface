#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import numpy as np
import time
import dlib
# start = time.time()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/hph/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        print ('TooManyFaces')
        return None
    if len(rects) == 0:
        print ('NoFaces')
        return None

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

SCALE_FACTOR=1
OUTER_POINTS = list(range(0,17))
OTHER_POINTS = list(range(61,67))
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    # OUTER_POINTS+OTHER_POINTS,
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

FEATHER_AMOUNT = 11
def transformation_from_points(points1, points2):
    """
        Return an affine transformation [s * R | T] such that:
            sum ||s*R*p1,i + T - p2,i||^2
        is minimized.
        """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

COLOUR_CORRECT_BLUR_FRAC = 0.6
# LEFT_EYE_POINTS = list(range(42, 48))
# RIGHT_EYE_POINTS = list(range(36, 42))

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += 128 * (im2_blur <= 1.0)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

if __name__ == '__main__':
    start =time.time()

    im1, landmarks1 = read_im_and_landmarks('../data/2.jpg')
    im2 = cv2.imread('../data/1.jpg')
    im2 = cv2.resize(im2,(im1.shape[1],im1.shape[0]))
    landmarks2 = get_landmarks(im2)
    # im2, landmarks2 = read_im_and_landmarks('../data/2.jpg')

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)
    cv2.imshow("fg",im2)
    cv2.imshow("bg",im1)

    cv2.imshow("mask",mask)
    cv2.imshow("warped_mask",warped_mask)
    cv2.imshow("combined_mask",combined_mask)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1.astype(np.float64), warped_im2.astype(np.float64), landmarks1)

    cv2.imshow("warped_corrected_im2",warped_corrected_im2.astype(np.uint8))

    points = landmarks1[ALIGN_POINTS].astype(np.uint8)
    c2 = np.mean(points, axis=0)
    c2 = np.array(c2)
    print (c2[0][0],c2[0][1])
    print(im1.shape, warped_corrected_im2.shape, combined_mask.shape)

    combined_mask=(combined_mask*255).astype(np.uint8)
    output_im = cv2.seamlessClone(warped_im2,im1,combined_mask,(int(c2[0][0]),int(c2[0][1])),cv2.NORMAL_CLONE )
    # print (output_im.shape)
    # output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imshow("merged",output_im.astype(np.uint8))
    print('spend ', time.time() - start, ' second')
    print('缺点是不能将眼睛和嘴巴很好地进行变换')
    cv2.waitKey()
