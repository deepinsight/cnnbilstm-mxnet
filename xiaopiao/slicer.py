#coding=utf-8
import glob
import numpy as np
import cv2
import sys

TARGETPREFIX = 'prepared/sliced/'

def Slicer(filename):
    img = cv2.imread(filename,0)
    cleanfilename = filename.split('/')[-1]
    hist1 = 255.0-np.mean(img, axis=1)
    hist1 = (hist1-hist1.min())/(255.0-hist1.min())

    THRESHOLD = 2.0/255.0
    HEADING = 7
    TRAILING = 1

    startpos = np.zeros(len(hist1))
    endpos = np.zeros(len(hist1))

    active = False

    for i in range(len(hist1)):
        if np.all(hist1[max(i-HEADING,0):i] > THRESHOLD) and active==False:
            startpos[max(i-HEADING,0)] = 1
            active = True
        if np.all(hist1[max(i-TRAILING,0):i] < THRESHOLD) and active==True:
            endpos[i] = 1
            active = False
     
    
    startpoint, endpoint = None, None
    pic_num = 0
    for i in range(len(startpos)):
        if startpos[i] == 1:
            startpoint = i
        if endpos[i] == 1:
            endpoint = i
            #
            pic_num+=1
            cv2.imwrite(TARGETPREFIX + cleanfilename + '{0:0>3}.jpg'.format(pic_num),img[startpoint:endpoint, :])
            startpoint, endpoint = None, None
    if startpoint != None:
        pic_num += 1
        cv.imwrite(TARGETPREFIX + cleanfilename + '{0:0>3}.jpg'.format(pic_num) ,img[startpoint:, :])
            
            
if __name__ == "__main__" :
    if len(sys.argv)==1:
        filelist = glob.glob('prepared/*.jpg')
        for filename in filelist:
            Slicer(filename)
    else:
        filename = sys.argv[1]
        Slicer(filename)
        
            

            
        

