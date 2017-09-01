#coding=utf-8

import os, sys, glob
import infer_ocr


filelist = glob.glob('xiaopiao/prepared/slicestded/*.jpg')

hit , total = 0, 0 

for filename in filelist:
    os.system("python infer_ocr.py "+filename)