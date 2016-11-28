#!/usr/bin/env python2

import os
import cv2
import openface
import copy
import numpy as np
import pickle
import time

start = time.time()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

align = openface.AlignDlib('/home/hph/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
net = openface.TorchNeuralNet('../models/nn4.v2.t7', 96, cuda=False)

imgDim = 96
def getRep(bgrImg, multiple=True):
    if bgrImg is None:
        raise Exception("Unable to load image: ")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    # cv2.imshow("1",rgbImg)
    # cv2.waitKey(1)
    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        return []

    reps = []
    for bb in bbs:
        alignedFace = align.align(
            imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {2}")

        rep = net.forward(alignedFace)
        reps.append((bb, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

class TestVideo(object):
    @classmethod
    def test_predict_video(cls, video_name,modeldir,compress_ratio = 0.5,multiple=True):
        with open(modeldir, 'r') as f:
            (le, clf) = pickle.load(f)
        cap = cv2.VideoCapture()
        cap.open(video_name)

        # width = cap.get(0)
        # height = cap.get(1)
        width = 1920
        height = 1080
        wrt = cv2.VideoWriter(video_name[:-4] + "_pred.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame_show = copy.deepcopy(frame)
            frame = cv2.resize(frame, (int(width*compress_ratio), int(height*compress_ratio)))

            reps = getRep(frame, multiple)
            if len(reps) > 1:
                print("List of faces in image from left to right")

            for r in reps:
                rep = r[1].reshape(1, -1)
                bb = r[0]

                if bb.left()<0:
                    l = 0
                elif bb.left()>width:
                    l=width-1
                else:
                    l=bb.left()

                if bb.right() < 0:
                    r = 0
                elif bb.right()>width:
                    r = width-1
                else:
                    r=bb.right()

                if bb.top() < 0:
                    t = 0
                elif bb.top()>height:
                    t = height-1
                else:
                    t = bb.top()

                if bb.bottom() < 0:
                    d = 0
                elif bb.bottom()>height:
                    d = height-1
                else:
                    d = bb.bottom()


                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]

                if confidence<0.7:
                    cv2.rectangle(frame, (l, t), (r, d), color=(255, 255, 0), thickness=4)
                    cv2.rectangle(frame_show, (int(l / compress_ratio), int(t / compress_ratio)),
                                  (int(r / compress_ratio), int(d / compress_ratio)),
                                  color=(255, 255, 0), thickness=4)
                else:
                    cv2.rectangle(frame, (l, t), (r, d), color=(0, 0, 255), thickness=4)
                    cv2.rectangle(frame_show, (int(l / compress_ratio), int(t / compress_ratio)),
                                  (int(r / compress_ratio), int(d / compress_ratio)),
                                  color=(0, 0, 0), thickness=4)
                if multiple:
                    print("Predict {} @ x={} with {:.2f} confidence.".format(person, bb, confidence))
                    if confidence<0.7:
                        cv2.putText(frame, "unknow.".format(person, confidence),
                                    (bb.center().x, bb.center().y), \
                                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 0, 0), 2)
                        cv2.putText(frame_show, "unknow.".format(person, confidence),
                                    (int(bb.center().x / compress_ratio), int(bb.center().y / compress_ratio)), \
                                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "{} @  {:.2f} confidence.".format(person,confidence),
                                    (bb.center().x,bb.center().y),\
                                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 2)
                        cv2.putText(frame_show, "{} @  {:.2f} confidence.".format(person, confidence),
                                    (int(bb.center().x/compress_ratio), int(bb.center().y/compress_ratio)), \
                                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 2)
                else:
                    print("Predict {} with {:.2f} confidence.".format(person, confidence))
                if isinstance(clf, LinearRegression):
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))

                cv2.imshow("1", frame)
                cv2.waitKey(1)
            wrt.write(frame_show)

if __name__ == '__main__':
    TestVideo.test_predict_video(video_name='../data/xiaoao1501.mp4', modeldir='../data/stars10-600/rep/classifierGaussianNB.pkl')
