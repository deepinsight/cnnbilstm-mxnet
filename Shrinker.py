#coding=utf-8

import glob
import cv2
import numpy as np
import re
import sys


def Shrinker(fromdir, todir, suffix='jpg'):
    filelist = glob.glob(fromdir+'*.'+suffix)
    for idx, filename in enumerate(filelist):
        cleanfilename = '.'.join(re.sub('-','',filename.split('/')[-1]).split('.')[:-1])
        if idx%10000==0:
            print idx
        img = cv2.imread(filename,0)
        hist0 = np.mean((img - 255.0)*(-1/255.0),axis=0)

        THRESHOLD = 5/255.0

        new_img = np.zeros(img.shape)+255.0

        j = 0
        for i in range(len(hist0)):
            if hist0[i] > THRESHOLD or i==0 or hist0[i-1] > THRESHOLD:
                new_img[:,j] = img[:,i]
                j+=1

        cv2.imwrite( todir+cleanfilename+'-.'+suffix, new_img)

if __name__ == "__main__":
    
    if len(sys.argv)!=3:
        print "python Shrinker.py FromDir ToDir"
    else:
        Shrinker(sys.argv[1],sys.argv[2])

