#coding=utf-8

import glob
import cv2
import numpy as np
import re
import sys

def Expander(fromdir, todir, suffix='jpg'):
    filelist = glob.glob(fromdir+'*.'+suffix)
    for idx, filename in enumerate(filelist):
        cleanfilename = '.'.join(re.sub('-','',filename.split('/')[-1]).split('.')[:-1])
        if idx%10000==0:
            print idx
        img = cv2.imread(filename,0)
        
        hist0 = np.mean((img - 255.0)*(-1/255.0),axis=0)
        hist1 = np.mean((img - 255.0)*(-1/255.0),axis=1)

        THRESHOLD = 10.0/255.0

        for up in range(0,img.shape[0]):
            if hist1[up]>THRESHOLD:
                break
        
        for down in range(img.shape[0],0,-1):
            if hist1[down-1]>THRESHOLD:
                break
        
        #down+=1
        
        for left in range(0,img.shape[1]):
            if hist0[left]>THRESHOLD:
                break
                
        for right in range(img.shape[1],0,-1):
            if hist0[right-1]>THRESHOLD:
                break
        
        #right+=1
        
        if right-left<4 or down-up<4:
            if idx%10000==0:
                print "small"
            cv2.imwrite(todir+cleanfilename+'.'+suffix, img)
        else:
            new_img = np.array(img[up:down,left:right])
            new_img = cv2.resize(new_img,(32,32))
            if idx%10000==0:
                print up,down,left,right, img.shape, "normal"
                print new_img.shape
            cv2.imwrite(todir+cleanfilename+'.'+suffix, new_img)

    
    
if __name__ == "__main__":
    
    if len(sys.argv)!=3:
        print "python Expander.py FromDir ToDir"
    else:
        Expander(sys.argv[1],sys.argv[2])