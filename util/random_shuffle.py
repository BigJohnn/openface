#-*- coding:utf-8 -*-
import os,shutil
import random

from PIL import Image

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

def random_choice(num):
    if not os.path.exists(RANDOM_CHOICED_ROOT):
        os.mkdir(RANDOM_CHOICED_ROOT)
    for clsdir in os.listdir(TRAIN_SOURCE):
        if not os.path.exists(RANDOM_CHOICED_ROOT+'/'+clsdir):
            os.mkdir(RANDOM_CHOICED_ROOT+'/'+clsdir+'/')
        print os.path.join(RANDOM_CHOICED_ROOT, clsdir)
        ls =  os.listdir(os.path.join(TRAIN_SOURCE,clsdir))
        slice = random.sample(ls,num)
        for item in slice:
            shutil.copy(os.path.join(TRAIN_SOURCE,clsdir,item),os.path.join(\
                        RANDOM_CHOICED_ROOT,clsdir,item))

if __name__ == '__main__':
    # random_shuffle()
    random_choice(500)

    pass
