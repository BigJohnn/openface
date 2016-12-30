#!/usr/bin/env python2
import cv2
import numpy as np
import os
import random
import shutil

import openface
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/hph/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        print 'TooManyFaces'
    if len(rects) == 0:
        print 'NoFaces'

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

OUTER_POINTS = list(range(0,17))
OTHER_POINTS = list(range(61,67))
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    OUTER_POINTS+OTHER_POINTS,
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
FEATHER_AMOUNT = 11
def restoreTextureFromTriangle(img,triple_pt):
    size=img.shape
    roi = (0, 0, size[1],size[0]+10)
    pts = np.float32([[triple_pt[0], triple_pt[1]], [triple_pt[2], triple_pt[3]], [triple_pt[4], triple_pt[5]]])
    warp_mat = cv2.getAffineTransform(pts,pts)
    # print warp_mat
    # warp_mat=np.array([[1,0,0],[0,1,0]])
    # warp_dst = np.zeros(img.shape, np.uint8)
    warp_final = np.zeros(img.shape, np.uint8)
    mask = np.zeros(img.shape, np.uint8)
    # orgRoi=img[roi]

    warp_dst = cv2.warpAffine(img,warp_mat,(size[0],size[1]))

    # warp_final[roi]=dstRoi
    mask = cv2.fillConvexPoly(mask,pts,1)
    warp_dst.copyTo(warp_final, mask)
    cv2.imshow("result", warp_final)
    cv2.waitKey()
    pass

def delaunay(points, img):
    size=img.shape
    adj=dict()
    indices=[]
    rect = (0, 0, size[1],size[0]+10)
    subdiv = cv2.Subdiv2D(rect)
    mappts = dict()
    for i,point in enumerate(points):
        try:
            mappts[point]
        except:
            mappts[point]=i
            subdiv.insert(point)
            pass
    edgeList = subdiv.getEdgeList()
    nE = len(edgeList)
    for e in edgeList:
        pt0=(e[0],e[1])
        pt1=(e[2],e[3])
        try:
            idx0=mappts[pt0]
            idx1=mappts[pt1]
        except:
            continue
        adj[(idx0,idx1)]=True
        adj[(idx1,idx0)]=True

    triangleList=subdiv.getTriangleList()
    triangleList = triangleList.tolist()
    nT=len(triangleList)

    for t in triangleList:
        pt0=(t[0],t[1])
        pt1=(t[2],t[3])
        pt2=(t[4],t[5])
        try:
            idx0=mappts[pt0]
            idx1=mappts[pt1]
            idx2=mappts[pt2]
            # restoreTextureFromTriangle(img,t)
        except:
            print 'Not a valid point#1'
            triangleList.remove(t)
            continue
        indices.append(idx0)
        indices.append(idx1)
        indices.append(idx2)
    return adj,indices, triangleList

def transformation_from_points(points1, points2):
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

COLOUR_CORRECT_BLUR_FRAC = 0.6
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))


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
    path = '../data/stars10/singles/baby.jpg'

    img = cv2.imread(path)

    # align = openface.AlignDlib('/home/hph/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
    # bb = align.getLargestFaceBoundingBox(img)
    # ls68 = openface.AlignDlib.findLandmarks(align, img, bb)
    #
    # for point in ls68:
    #     cv2.circle(img, (point[0], point[1]), 2, (255, 255, 0))
    #
    #
    # adj, indices, triangleList= delaunay(ls68,img)

    landmarks = get_landmarks(img)
    mask = get_face_mask(img, landmarks)
    # pts = np.float32([[1,1], [1,1], [2,2]])
    # M = cv2.getAffineTransform(pts, pts)
    M = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0,0.0,1.0]])
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
    # M=transformation_from_points(landmarks[ALIGN_POINTS],landmarks[ALIGN_POINTS])
    warped_mask = warp_im(mask, M, img.shape)
    combined_mask = np.max([get_face_mask(img, landmarks), warped_mask],
                              axis=0)
    # output_im = img * (1.0 - mask)+ img * mask
    output_im= img*mask

    cv2.imshow("1",output_im.astype(np.uint8))
    cv2.waitKey()
    pass


