import os
import cv2
import openface
import copy
import numpy as np
import pickle
import time

start = time.time()

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

align = openface.AlignDlib('models/dlib/shape_predictor_68_face_landmarks.dat')
net = openface.TorchNeuralNet('models/nn4.v2.t7', 96, cuda=True)

imgDim = 96
def getRep(imgPath, multiple=False):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face: {}".format(imgPath))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))

        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def infer(args, multiple=False):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    for img in args.imgs:
        print("\n=== {} ===".format(img))
        reps = getRep(img, multiple)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]

            if multiple:
                print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                         confidence))
            else:
                print("Predict {} with {:.2f} confidence.".format(person, confidence))
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))

class TestVideo(object):
    @classmethod
    def test_predict_video(cls, video_name, dets_file,multiple=False):
        with open('models/classifier10stars.pkl', 'r') as f:
            (le, clf) = pickle.load(f)
        cap = cv2.VideoCapture()
        cap.open(video_name)
        wrt = cv2.VideoWriter(video_name[:-4] + "_pred.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (576, 432))

        pred_results = []
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            # frame = cv2.resize(frame, (576, 432))
            frame_show = copy.deepcopy(frame)
            print("\n=== {} ===".format(frame))
            reps = getRep(frame, multiple)
            if len(reps) > 1:
                print("List of faces in image from left to right")
            for r in reps:
                rep = r[1].reshape(1, -1)
                bbx = r[0]
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]

                if multiple:
                    print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                             confidence))
                else:
                    print("Predict {} with {:.2f} confidence.".format(person, confidence))
                if isinstance(clf, GMM):
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))
            wrt.write(frame_show)
            # cv2.imshow("out", frame_show)
            # cv2.waitKey(0)
        #     pred_results.append(preds)
        #     i += 1
        #
        # print "len(pred_results)=", len(pred_results)


if __name__ == '__main__':
    TestVideo.test_predict_video('../Data/wumn.flv', '../Models/predict/classifier10stars.pkl')