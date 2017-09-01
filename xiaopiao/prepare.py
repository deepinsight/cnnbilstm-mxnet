#coding=utf-8
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob

def Resizer( filename , narrower = True):
    print filename
    
    img = cv2.imread(filename,0)
    
    print img.shape
    
    if narrower:
        shape = img.shape[0],img.shape[1]/2
    else:
        shape = img.shape
        
    newimg = cv2.resize(img, (shape[1],shape[0]) )
    
    print newimg.shape
    
    padshape = max(shape[0],600),max(shape[1],600)
    
    print padshape
    
    padimg = np.zeros(padshape) + 255.0
    
    print padimg.shape
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            padimg[i][j] = newimg[i][j]
    
    #im = Image.new(mode='RGBA', size=newshape)
    #im.paste(newimg,)
    
    newfilename = './prepared/' + ('narrowd_' if narrower else 'resized_') + filename
    cv2.imwrite(newfilename, padimg)
    
if __name__ == "__main__" :
    fileList = glob.glob("./*.jpg")
    for filename in fileList:
        cleanfilename = filename.split("/")[-1]
        Resizer(cleanfilename,False)
        Resizer(cleanfilename,True)