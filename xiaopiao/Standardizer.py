#coding=utf-8
import glob
import numpy as np
import cv2
import sys

FROMPREFIX = 'prepared/slicetagged/'
TOPREFIX = 'prepared/slicestded/'

def Standardizer(filename):
    #filename = 'prepared/slicetagged/1_收银员：806---七店前台.jpg'

    img = cv2.imread(filename,0)
    cleanfilename = filename.split('/')[-1]
    targetfilename = TOPREFIX+cleanfilename
    
    if img==None:
        return None
    
    hist0 = 255.0 - np.mean(img, axis=0)
    hist0 = (hist0-hist0.min())/(255.0-hist0.min())
    THRESHOLD = 2.0/255.0
    '''
    if cleanfilename == '1_------------------.jpg':
        for i in range(len(hist0)/8):
            print hist0[8*i:8*(i+1)]
    '''
    #startpos = 
    #endpos = 
    for i in range(len(hist0)):
        if hist0[i]<=THRESHOLD:
            continue
        else:
            break
    for j in range(len(hist0)):
        if hist0[-(j+1)]<=THRESHOLD:
            continue
        else:
            break

    tailoredimg = img[:,i:(len(hist0)+1-j)]
    t_shape = tailoredimg.shape
    # rtp = np.zeros((30, 900)) + 255.0
    #print 1.0*t_shape[1]/t_shape[0]
    if t_shape[1]>30*t_shape[0]: # if False:
        newwidth = 900
        newheight = np.int(900.0/t_shape[1]*t_shape[0])
        #print newheight
        rt = cv2.resize(tailoredimg, (newwidth,newheight))
    else:
        newheight = 30
        newwidth = np.int(30.0/t_shape[0]*t_shape[1])
        #print newwidth
        rt = cv2.resize(tailoredimg, (newwidth,newheight))
    
    rtp = np.zeros((30, 900 )) + 255.0
    
    heightoffset = (30-newheight)/2
    
    for i in range(newheight):
        for j in range(newwidth):
            rtp[i+heightoffset,j] = rt[i,j]
        
    cv2.imwrite( targetfilename , rtp)
    
    
if __name__ == "__main__":
    filelist = glob.glob(FROMPREFIX+'*.jpg')
    for filename in filelist:
        #print filename
        Standardizer(filename)