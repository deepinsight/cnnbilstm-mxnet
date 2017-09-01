#coding=utf-8

import cv2
import numpy as np
import os,sys
import glob




def Filter(targetpathprefix, filename, scale, method):
    """
    scale:1~16
    method:blur,GaussianBlur
    """
    cleanfilename = filename.split('/')[-1]
    filepathprefix = '/'.join(filename.split('/')[0:-1])
    if len(cleanfilename.split('_'))!=2:
        print cleanfilename
    cleanfilenamelabel,cleanfilenamepart = cleanfilename.split('_')
    scalestr = '{0:0>2}'.format(scale)
    methodstr = '0' if method == "blur" else "1"
    randstr = '{0:0>4}'.format(np.random.randint(0,10000))
    newlabel = randstr + cleanfilenamelabel + scalestr + methodstr
    
    if method == "blur":
        blurmethod = lambda img,scale: cv2.blur(img,(scale,scale))
    elif method == "GaussianBlur":
        blurmethod = lambda img,scale: cv2.GaussianBlur(img,(scale,scale),0)
        
    img = cv2.imread(filename,0)
    if method=='GaussianBlur' and scale%2==0:
        #print "Size: ",scale  
        return None
    bimg = blurmethod(img, scale)
    newfilename = targetpathprefix+'/'+newlabel+'_'+cleanfilenamepart
    print newfilename
    cv2.imwrite(newfilename,bimg)
      
    


if __name__ == "__main__":
    
    filelist = glob.glob("prepared/slicestded/*.jpg")
    #filelist = ['prepared/slicestded/1_1.请保留好您的购物凭证，凭此单换发票。---.jpg']
    if len(sys.argv)==1:
        repeat = 1
    else:
        repeat = int(sys.argv[1])

    
    for filename in filelist:
        for scale in range(1,17):
            for method in ['GaussianBlur']:
                for i in range(repeat):
                    Filter('/data2/img_data/train_data_varlen_5_45_CHLang_ConcatMix', filename, scale, method)
                    
        
        
        
        
        
        