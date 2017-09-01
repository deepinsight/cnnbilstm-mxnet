#coding=utf-8
import sys
import glob
sys.path.insert(0, "/home/liang/frankwang/mxnet/python")
import numpy as np
import mxnet as mx

from lstm_model import LSTMInferenceModel
from han_gen_langmodel import *

import cv2, random


BATCH_SIZE = 1
SEQ_LENGTH = 300


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret


def gen_rand():
    buf = ""
    max_len = random.randint(3,4)
    for i in range(max_len):
        buf += str(random.randint(0,9))
    return buf

if __name__ == '__main__':
    
    if len(sys.argv)==1:
        #filelist = glob.glob('xiaopiao/prepared/slicestded/*.jpg')
        #filelist = ['xiaopiao/prepared/slicestded/1_付款：---礼券：5.00，储值-----.jpg']
        filelist = ['xiaopiao/prepared/slicestded/1_1.请保留好您的购物凭证，凭此单换发票。---.jpg']
    else:
        filelist = [sys.argv[1]]
        

    num_hidden = 1024
    num_lstm_layer = 2

    prefix = 'mymodel/MIXLANGUAGE_BATCH128_SEQ300_varlen_1024-2-0.0003-100-mmt0.9-predhan'
    num_epoch = 11
    learning_rate = 0.0005
    momentum = 0.9
    num_label = 45

    n_channel = 1
    contexts = [mx.context.gpu(1)]
    _, arg_params, __ = mx.model.load_checkpoint( prefix , num_epoch)

    LARGEST_WIDTH = 150
    LARGEST_HEIGHT = 32
    
    for filename in filelist:
        print filename
        img = cv2.imread(filename ,0)
        img = img.transpose(1, 0)
    
        """
        img = img.reshape((1,150*32))
        img = np.multiply(img, 1 / 255.0)



        sliced_images = []
        sliced_images.append(img)
    
        """
        sliced_images = []
    
        for i in range(img.shape[0]/LARGEST_WIDTH):
            part_img = img[i*LARGEST_WIDTH:(i+1)*LARGEST_WIDTH,:]
            new_img = np.zeros((LARGEST_WIDTH,LARGEST_HEIGHT))+255.0
            for k in range(LARGEST_WIDTH):
                for j in range(30):
                    new_img[k,j+(LARGEST_HEIGHT-30)/2] = part_img[k,j]
    
    
            new_img = new_img.reshape((1, LARGEST_WIDTH * LARGEST_HEIGHT))
            new_img = np.multiply(new_img, 1 / 255.0)
            print new_img.shape
            sliced_images.append(new_img)
    
    
        print "Image Loaded."
    
        data_shape = [('data', (1, n_channel * LARGEST_WIDTH * LARGEST_HEIGHT))]
        input_shapes = dict(data_shape)



        p = []
        for img in sliced_images:
            print "Rebuild Inference..."
            model = LSTMInferenceModel(num_lstm_layer,
                               SEQ_LENGTH,
                               num_hidden=num_hidden,
                               num_label=num_label,
                               arg_params=arg_params,
                               data_size = n_channel * LARGEST_HEIGHT * LARGEST_WIDTH,
                               ctx=contexts[0])

            print "Rebuilded."
            #print "Forwarding..."
            prob = model.forward(mx.nd.array(img))
            #print "Forwarded."
            for k in range(SEQ_LENGTH):
                p.append(np.argmax(prob[k]))

        #p = ctc_label(p)
        print 'Predicted label: ' + str(p)

        print 'Predicted number: ' + ''.join(get_inverse_label(p))
        
        del p