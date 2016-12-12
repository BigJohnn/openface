#-*- coding:utf-8 -*-
import os,shutil
import random
import cv2
import re
import argparse
import time
import operator
import rtsa

DATA_ROOT = '../Data/costars'
# RANDOM_CHOICED_ROOT ='../Data/rcdata'
RANDOM_CHOICED_ROOT ='../Data/ultimate_train'
# TRAIN_SOURCE = '../Data/train_data'
TRAIN_SOURCE = '../Data/rcdata'

class image():
    def __init__(self):
        self.id=-1
        self.path = ''
def random_shuffle():
    imagelists=[]

    for dir in os.listdir(DATA_ROOT):
        pref = os.path.join(DATA_ROOT, dir)
        images = []
        cid = -1
        for item in os.listdir(os.path.join(DATA_ROOT,dir)):
            im = image()
            cid = cid+1
            im.id = cid
            im.path = os.path.join(pref,item)
            images.append(im)
        imagelists.append(images)
    counts = []
    for item in imagelists:
        counts.append(len(item))
    sortedcounts = sorted(counts,reverse=True)
    newlists = []
    for item,j in enumerate(counts):
        rs = range(0,sortedcounts[0])
        random.shuffle(rs)
        for ind,i in enumerate(rs):
            rs[ind] = i%j
        newlists.append(rs)
    pass

def random_choice(args):
    if not os.path.exists(args.inDir):
        print "InputDir not exists!"
        return
    if not os.path.exists(args.outDir):
        os.mkdir(args.outDir)
    for clsdir in os.listdir(args.inDir):
        if not os.path.exists(args.outDir+'/'+clsdir):
            os.mkdir(args.outDir+'/'+clsdir+'/')
        print os.path.join(args.outDir, clsdir)
        ls =  os.listdir(os.path.join(args.inDir,clsdir))
        if len(ls)<=args.num:
            slice=ls
        else:
            slice = random.sample(ls,args.num)
        for item in slice:
            shutil.copy(os.path.join(args.inDir,clsdir,item),os.path.join( \
                args.outDir,clsdir,item))

def colorclahe(img):
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(2,(10,10))
    cb = clahe.apply(b)
    cg = clahe.apply(g)
    cr = clahe.apply(r)
    return cv2.merge([cb,cg,cr])

ALIGNED_REGEN = '../data/stars10-600/aligned-regen'
ALIGNED = '../data/stars10-600/aligned'
def regenerate():
    if not os.path.exists(ALIGNED_REGEN):
        os.mkdir(ALIGNED_REGEN)
    for clsdir in os.listdir(ALIGNED):
        if not re.search('\.t7', clsdir):
            if not os.path.exists(ALIGNED_REGEN+'/'+clsdir):
                os.mkdir(ALIGNED_REGEN+'/'+clsdir+'/')
            for item in os.listdir(ALIGNED+'/'+clsdir):
                img = cv2.imread(os.path.join(ALIGNED+'/'+clsdir,item))
                cv2.imwrite(os.path.join(ALIGNED_REGEN+'/'+clsdir,item),colorclahe(img))

def split_train_test(args):
    start = time.time()
    destPath = args.logdir

    fullFaceDirectory = [os.path.join(args.data, f) for f in os.listdir(
        args.data)]

    noOfImages = []
    folderNames = []

    for folder in fullFaceDirectory:
        try:
            ls = os.listdir(folder)
            noOfImages.append(len(ls))
            folderName = folder.split('/')[-1:][0]
            folderNames.append(folderName)
            os.mkdir(os.path.join(args.train,folderName))
            os.mkdir(os.path.join(args.test, folderName))
            trainls = random.sample(ls,int(len(ls)*args.ratio))
            testls = list(set(ls).difference(set(trainls)))
            for f in trainls:
                shutil.copy(os.path.join(folder,f),os.path.join(args.train,folderName))
            for f in testls:
                shutil.copy(os.path.join(folder,f),os.path.join(args.test, folderName))
            print folder.split('/')[-1:][0] +": " +\
            str(len(os.listdir(folder)))
        except:
            pass

    # Sorting
    noOfImages_sorted, folderNames_sorted = zip(
        *sorted(zip(noOfImages, folderNames), key=operator.itemgetter(0), reverse=True))

    with open(os.path.join(destPath, "List_of_folders_and_number_of_images.txt"), "w") as text_file:
        for f, n in zip(folderNames_sorted, noOfImages_sorted):
            text_file.write("{} : {} \n".format(f, n))
    if args.verbose:
        print "Sorting lfw dataset took {} seconds.".format(time.time() - start)

def ifSmallThenGenerate(args):
    for dir in os.listdir(args.root):
        folder = os.path.join(args.root,dir)
        while len(os.listdir(folder))<args.th:
            rtsa.data_aug(folder)
        verbose = random.sample(os.listdir(folder),len(os.listdir(folder))-args.th)
        # print verbose
        for item in verbose:
            print os.path.join(args.root,dir,item)
            os.remove(os.path.join(args.root,dir,item))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Gogogo',
                                     description='randomchoice')
    parser.add_argument('--verbose', action='store_true',help='Time consume.')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    rcParser = subparsers.add_parser('ranch',
                                        help="Random choice some items.")
    rcParser.add_argument(
        '--inDir', type=str,
        help='Input Dir.')
    rcParser.add_argument(
        '--num', type=int,default=600,
        help='To pick how many items.')
    rcParser.add_argument('--outDir',
                        type=str,
                        help='Output Dir.')

    splitParser = subparsers.add_parser('split',
                                        help="Split original dataset into train set and test set.")
    splitParser.add_argument(
        '--ratio', type=float,
        default=0.9,help='The ratio of train samples out of all.'
    )
    splitParser.add_argument(
        '--data', type=str,
        help='Data root dir.')

    #Default values for this triple is set later.
    splitParser.add_argument(
        '--train', type=str,
        help='Train data dir, no need to provide.')
    splitParser.add_argument(
        '--test', type=str,
        help='Test data dir, no need to provide.')
    splitParser.add_argument(
        '--logdir', type=str,
        help='Log dir, no need to provide.')
    mgParser = subparsers.add_parser('maybegen',
                                     help="Conditionaly generate some new data to train with.")
    mgParser.add_argument(
        '--root', type=str, help='Data root, which contains many folders of images.'
    )
    mgParser.add_argument(
        '--th', type=int, default=540, help='The threshold.'
    )
    args = parser.parse_args()


    if args.mode=='ranch':
        print args.inDir
        print args.num
        print args.outDir
        random_choice(args)
    elif args.mode=='split':
        if args.train==None:
            args.train=os.path.join(args.data,'../train')
        if args.test==None:
            args.test = os.path.join(args.data,'../test')
        if args.logdir==None:
            args.logdir = os.path.join(args.data,'..')
        print 'turtle'
        print args.ratio*100,'%'
        print args.train, args.test,args.logdir
        split_train_test(args)
    elif args.mode=='maybegen':
        ifSmallThenGenerate(args)
    else:
        pass
    # random_shuffle()
    # regenerate()

    # random_choice(500)

    pass
