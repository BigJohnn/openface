#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import openface
import copy
import numpy as np
import pickle
import time
import re
from scipy import misc
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# # from sklearn.grid_search import GridSearchCV
# from sklearn.mixture import GMM
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
import operator


import sys
sys.path.append("/home/hph/openface/util")

import tensorflow as tf
import detect_face

start = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Compare(object):
    def __init__(self, model_dir, image_size=160, margin=44, gpu_memory_fraction=0.3):
        self.image_size = image_size
        self.margin = margin
        self.gpu_memory_fraction = gpu_memory_fraction
        self.model_dir = model_dir

        # Load the model
        print('Model directory: %s' % self.model_dir)
        meta_file, ckpt_file = self.__get_model_filenames()
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        print('Creating networks and loading parameters...')
        with tf.Graph().as_default():
            self.__sess_feature = tf.Session()
            self.__load_model(meta_file, ckpt_file)

            # Get input and output tensors
            self.__images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.__embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction,allow_growth = True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    log_device_placement=False,
                                                    allow_soft_placement=True
                                                    ))
            with sess.as_default():
                self.__pnet, self.__rnet, self.__onet = detect_face.create_mtcnn(sess, self.model_dir)

    def __load_model(self, meta_file, ckpt_file):
        model_dir_exp = os.path.expanduser(self.model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
        saver.restore(self.__sess_feature, os.path.join(model_dir_exp, ckpt_file))

    def eval_dist(self, image1, image2):
        init_imagelist = [image1, image2]
        images = self.load_and_align_data(init_imagelist)
        result = self.__extract_calculate(images)
        # result = self.__extract_calculate(np.stack(init_imagelist))
        return result

    def load_and_align_data(self, init_images):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        t01 = time.clock()
        nrof_samples = len(init_images)
        img_list = [None] * nrof_samples
        for i in range(nrof_samples):
            # init_images[i] = cv2.resize(init_images[i], (self.image_size, self.image_size))
            #
            # pre_whitened = self.__pre_whiten(init_images[i])

            img_size = np.asarray(init_images[i].shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(init_images[i], minsize,
                                                        self.__pnet, self.__rnet, self.__onet,
                                                        threshold, factor)
            # print (bounding_boxes.shape)
            if np.size(bounding_boxes) == 0:
                det = np.squeeze(np.array([0,0,img_size[1],img_size[0]]))
            else:
                det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
            cropped = init_images[i][bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
            pre_whitened = self.__pre_whiten(aligned)
            img_list[i] = pre_whitened
        images = np.stack(img_list)
        t02 = time.clock()
        # print("align time for 2 images: %fs" % (t02-t01))
        return images

    def __extract_calculate(self, images):
        # Run forward pass to calculate embeddings
        feed_dict = {self.__images_placeholder: images}
        t03 = time.clock()
        emb = self.__sess_feature.run(self.__embeddings, feed_dict=feed_dict)
        t04 = time.clock()
        # print("extract feature time: %fs" % (t04-t03))
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))

        return dist

    @staticmethod
    def __pre_whiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def __get_model_filenames(self):
        files = os.listdir(self.model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % self.model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % self.model_dir)
        meta_file = meta_files[0]
        ckpt_files = [s for s in files if 'ckpt' in s]
        if len(ckpt_files) == 0:
            raise ValueError('No checkpoint file found in the model directory (%s)' % self.model_dir)
        elif len(ckpt_files) == 1:
            ckpt_file = ckpt_files[0]
        else:
            ckpt_iter = [(s, int(s.split('-')[-1])) for s in ckpt_files if 'ckpt' in s]
            sorted_iter = sorted(ckpt_iter, key=lambda tup: tup[1])
            ckpt_file = sorted_iter[-1][0]
        return meta_file, ckpt_file

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

class image():
    def __init__(self):
        self.src = []
        self.name = ''
        self.dist = 10000.0

class TestVideo(object):
    @classmethod
    def test_predict_video(cls, video_name,modeldir, compress_ratio = 0.9,multiple=True):
        with open(modeldir, 'r') as f:
            (le, clf) = pickle.load(f)
        cap = cv2.VideoCapture()
        cap.open(video_name)

        width = cap.get(0)
        height = cap.get(1)
        width = 1104
        height = 622
        # print video_name[:-4]
        wrt = cv2.VideoWriter(video_name[:-4] + "_pred.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height)) #image size must == (width, height)

        positive_clss = 'baby ch cqn dc fbb hg hjh lxr md wzl'
        while cap.isOpened():

            ret, frame = cap.read()
            if frame is None:
                break
            frame_write = copy.deepcopy(frame)
            frame = cv2.resize(frame, (int(width*compress_ratio), int(height*compress_ratio)))

            reps = getRep(frame, multiple)
            if len(reps) > 1:
                print("List of faces in image from left to right")
            t01 = time.clock()
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

                thresh = 0.7
                if confidence<thresh:
                    cv2.rectangle(frame, (l, t), (r, d), color=(255, 255, 0), thickness=4)
                    cv2.rectangle(frame_write, (int(l / compress_ratio), int(t / compress_ratio)),
                                  (int(r / compress_ratio), int(d / compress_ratio)),
                                  color=(255, 255, 0), thickness=4)
                else:
                    cv2.rectangle(frame, (l, t), (r, d), color=(0, 0, 255), thickness=4)
                    cv2.rectangle(frame_write, (int(l / compress_ratio), int(t / compress_ratio)),
                                  (int(r / compress_ratio), int(d / compress_ratio)),
                                  color=(0, 0, 0), thickness=4)
                if multiple:
                    print("Predict {} @ x={} with {:.2f} confidence.".format(person, bb, confidence))
                    if confidence<thresh or not re.search(person,positive_clss):
                        cv2.putText(frame, "unknow.".format(person, confidence),
                                    (bb.center().x, bb.center().y), \
                                    cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 0, 0), 2)
                        cv2.putText(frame_write, "unknow.".format(person, confidence),

                                    (int(bb.center().x / compress_ratio), int(bb.center().y / compress_ratio)), \
                                    cv2.FONT_HERSHEY_PLAIN, 1.1/(1+compress_ratio), (255, 0, 0), int(2/(1+compress_ratio)))
                    else:
                        cv2.putText(frame, "{} @  {:.2f} confidence.".format(person,confidence),
                                    (bb.center().x,bb.center().y),\
                                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 2)
                        cv2.putText(frame_write, "{} @  {:.2f} confidence.".format(person, confidence),
                                    (int(bb.center().x/compress_ratio), int(bb.center().y/compress_ratio)), \
                                    cv2.FONT_HERSHEY_PLAIN, 1.1/(1+compress_ratio), (0, 255, 255), int(2/(1+compress_ratio)))
                else:
                    print("Predict {} with {:.2f} confidence.".format(person, confidence))
                if isinstance(clf, LinearRegression):
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))

                cv2.imshow("1", frame)
                cv2.waitKey(1)
            wrt.write(frame_write)
            t02 = time.clock()
            # print (t02 - t01)

    @classmethod
    def cmp_predict_video(cls, video_name, cmp, canonical_imgs, compress_ratio=0.9, multiple=True):
        cap = cv2.VideoCapture()
        cap.open(video_name)

        # width = cap.get(0)
        # height = cap.get(1)
        width = 480
        height = 360
        # print video_name[:-4]
        wrt = cv2.VideoWriter(video_name[:-4] + "_pred.avi", cv2.VideoWriter_fourcc(*'XVID'), 20,
                              (width, height))  # image size must == (width, height)

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame_write = copy.deepcopy(frame)
            frame = cv2.resize(frame, (int(width * compress_ratio), int(height * compress_ratio)))

            reps = getRep(frame, multiple)
            if len(reps) > 1:
                print("List of faces in image from left to right")
            t01 = time.clock()
            for r in reps:
                # rep = r[1].reshape(1, -1)
                bb = r[0]

                if bb.left() < 0:
                    l = 0
                elif bb.left() > width:
                    l = width - 1
                else:
                    l = bb.left()

                if bb.right() < 0:
                    r = 0
                elif bb.right() > width:
                    r = width - 1
                else:
                    r = bb.right()

                if bb.top() < 0:
                    t = 0
                elif bb.top() > height:
                    t = height - 1
                else:
                    t = bb.top()

                if bb.bottom() < 0:
                    d = 0
                elif bb.bottom() > height:
                    d = height - 1
                else:
                    d = bb.bottom()
                roi = frame[t:d,l:r]
                if (roi.shape[0]<=0) or (roi.shape[1]<=0):
                    print (l,r,t,d)
                    continue
                # print(roi.shape)
                roi = cv2.resize(roi,(imgDim,imgDim))
                cv2.imshow('roi', roi)
                cv2.waitKey(1)
                for iter,item in enumerate(canonical_imgs):
                    canonical_imgs[iter].dist = cmp.eval_dist(roi, item.src)

                cmpfun = operator.attrgetter('dist') # sort by dist
                canonical_imgs.sort(key=cmpfun)

                distance = canonical_imgs[0].dist
                person = canonical_imgs[0].name

                thresh = 1.01
                if distance < thresh:
                    cv2.rectangle(frame, (l, t), (r, d), color=(255, 255, 0), thickness=4)
                    cv2.rectangle(frame_write, (int(l / compress_ratio), int(t / compress_ratio)),
                                  (int(r / compress_ratio), int(d / compress_ratio)),
                                  color=(255, 255, 0), thickness=4)
                else:
                    cv2.rectangle(frame, (l, t), (r, d), color=(0, 0, 255), thickness=4)
                    cv2.rectangle(frame_write, (int(l / compress_ratio), int(t / compress_ratio)),
                                  (int(r / compress_ratio), int(d / compress_ratio)),
                                  color=(0, 0, 0), thickness=4)
                print("Predict {} @ x={} with {:.2f} distance.".format(person, bb, distance))
                if distance > thresh:
                    cv2.putText(frame, "unknow.".format(person, distance),
                                (bb.center().x, bb.center().y), \
                                cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 0, 0), 2)
                    cv2.putText(frame_write, "unknow.".format(person, distance),

                                (int(bb.center().x / compress_ratio), int(bb.center().y / compress_ratio)), \
                                cv2.FONT_HERSHEY_PLAIN, 1.1 / (1 + compress_ratio), (255, 0, 0),
                                int(2 / (1 + compress_ratio)))
                else:
                    cv2.putText(frame, "{} @  {:.2f} distance.".format(person, distance),
                                (bb.center().x, bb.center().y), \
                                cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 255), 2)
                    cv2.putText(frame_write, "{} @  {:.2f} distance.".format(person, distance),
                                (int(bb.center().x / compress_ratio), int(bb.center().y / compress_ratio)), \
                                cv2.FONT_HERSHEY_PLAIN, 1.1 / (1 + compress_ratio), (0, 255, 255),
                                int(2 / (1 + compress_ratio)))

                cv2.imshow("1", frame)
                cv2.waitKey(1)

            wrt.write(frame_write)
            t02 = time.clock()
            print (t02 - t01)

    @classmethod
    def predict_pn(cls, modeldir, posisampledir, negasampledir):
        with open(modeldir, 'r') as f:
            (le, clf) = pickle.load(f)

        positive_clss = 'baby ch cqn dc fbb hg hjh lxr md wzl'
        # # thresh = 0.7
        # tpr = []
        tpr = []
        fpr = []
        accuracy = []
        for thresh in range(10):
            thresh = thresh / 10.0
            p = 0
            n = 0
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for dir in os.listdir(posisampledir):
                if re.search('\.',dir):
                    continue
                p+=1
                for item in os.listdir(os.path.join(posisampledir,dir)):
                    img = cv2.imread(os.path.join(posisampledir,dir,item))
                    rep = net.forward(img)
                    rep.reshape(1,-1)
                    predictions = clf.predict_proba(rep).ravel()
                    maxI = np.argmax(predictions)
                    person = le.inverse_transform(maxI)
                    confidence = predictions[maxI]

                    if confidence>thresh and re.search(person,positive_clss):
                        tp+=1
                    else:
                        fn+=1
                print ('p:',p)

            for dir in os.listdir(negasampledir):
                if re.search('\.',dir):
                    continue
                n+=1
                for item in os.listdir(os.path.join(negasampledir,dir)):
                    img = cv2.imread(os.path.join(negasampledir, dir, item))
                    rep = net.forward(img)
                    rep.reshape(-1,1)
                    predictions = clf.predict_proba(rep).ravel()
                    maxI = np.argmax(predictions)
                    person = le.inverse_transform(maxI)
                    confidence = predictions[maxI]

                    if confidence > thresh and re.search(person,positive_clss):
                        fp += 1
                    else:
                        tn += 1
                print ('n:',n)
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
        accuracy.append((tp+tn)/(p+n))

        print ('tpr:',tpr,'fpr:',fpr,'accuracy:',accuracy)


if __name__ == '__main__':
    # TestVideo.test_predict_video(video_name='../data/bpb150515.flv', modeldir='../data/stars10mix/rep/classifierLinearSvm23a900.pkl',compress_ratio=0.6)

    TestVideo.predict_pn('../data/stars10mix/rep/classifierLinearSvm23a900.pkl','../data/stars10-600/aligned','../aligned-11-16')
    # I = cv2.imread('../data/stars10/singles/baby.jpg')
    #
    # x = 'dsds'
    # print (x)
    # dst = cv2.resize(I,(100,100))
    # cv2.imshow('1',dst)
    # cv2.waitKey()

    # start = time.clock()
    # cp = Compare('compare/models', image_size=160, margin=44, gpu_memory_fraction=0.6)
    # end = time.clock()
    # print ('init time',end-start,'s')
    # images = []
    # pref = '../data/stars10/sinalin'
    # for dir in os.listdir(pref):
    #     im = image()
    #     im.src = cv2.imread(os.path.join(pref, dir))
    #     im.name = dir.split('.')[0]
    #     images.append(im)
    #
    # TestVideo.cmp_predict_video(video_name='../data/xiaoao1501.avi', cmp = cp, canonical_imgs=images, multiple=True)
