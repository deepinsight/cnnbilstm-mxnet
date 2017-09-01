#coding=utf-8
import sys

sys.path.insert(0, "/home/liang/frankwang/mxnet/python")
import numpy as np
import mxnet as mx

from lstm_model import LSTMInferenceModel
from han_gen_langmodel import *

import cv2, random, glob

BATCH_SIZE = 1
SEQ_LENGTH = 300

def get_padded_label(s,cutoff=45):
    out = np.zeros(cutoff)
    beforepad = get_label(s)
    #print len(s),len(out)
    curpos = -1
    for i in range(min(len(s),cutoff)):
        if i>0 and beforepad[i]==3562 and beforepad[i-1]==3562:
            pass
        else:
            curpos+=1
            out[curpos] = beforepad[i]

        #out[i] = beforepad[i]
    return out

def LCS(p,l):
    if len(p)==0 or len(l)==0:
        return 0
    P = np.array(list(p)).reshape((1,len(p)))
    L = np.array(list(l)).reshape((len(l),1))
    M = np.int32(P==L)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            up = 0 if i==0 else M[i-1,j]
            left = 0 if j==0 else M[i,j-1]
            M[i,j] = max(up,left, M[i,j] if (i==0 or j==0) else M[i,j]+M[i-1,j-1])
    return M.max()


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

def remove_trailing_zero(l):
    i = 0
    #print l
    for i in range(len(l)-1):
        if l[-i-1]==0 and l[-i-2]==0:
            continue
        else:
            break
    if i == 0:
        return l
    else:
        return l[:-i]

def gen_rand():
    buf = ""
    max_len = random.randint(3,4)
    for i in range(max_len):
        buf += str(random.randint(0,9))
    return buf


def infer_main(filelist):

    num_hidden = 1024
    num_lstm_layer = 2

    prefix = 'mymodel/MIXLANGUAGE_BATCH128_SEQ300_varlen_1024-2-0.0003-100-mmt0.9-predhan'
    num_epoch = 15
    learning_rate = 0.0005
    momentum = 0.9
    num_label = 45

    n_channel = 1
    contexts = [mx.context.gpu(5)]
    _, arg_params, __ = mx.model.load_checkpoint( prefix , num_epoch)

    model = LSTMInferenceModel(num_lstm_layer,
                          SEQ_LENGTH,
                          num_hidden=num_hidden,
                          num_label=num_label,
                          arg_params=arg_params,
                          data_size = n_channel * 30 * 900,
                          ctx=contexts[0])


    for filename in filelist:

        print filename

        label = filename.split('/')[-1].split('_')[1][:-4].decode('utf-8')

        truelabel = np.int64(remove_trailing_zero(get_padded_label(label)))

        img = cv2.imread(filename ,0)
        img = img.transpose(1, 0)
        img = img.reshape((1, 900 * 30))
        img = np.multiply(img, 1 / 255.0)

        print "Image Loaded."

        data_shape = [('data', (1, n_channel * 900 * 30))]
        input_shapes = dict(data_shape)

        prob = model.forward(mx.nd.array(img))

        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(prob[k]))

        p = ctc_label(p)

        print "True label:", truelabel
        hit = LCS(p, truelabel)

        print 'Predicted label: ' , str(p)
        print 'Predicted string: ' , u''.join(get_inverse_label([item for item in p if item>0])).encode('utf-8')
        print 'Hit Detail: ', hit, "in ", len(truelabel)


if __name__ == '__main__':

    """
    if len(sys.argv)==1:
        filename = 'xiaopiao/prepared/slicestded/0_1.此单为退换货唯一凭证，盖章有效妥善保管。.jpg'
    else:
        filename = sys.argv[1]
    """

    if len(sys.argv)==1:
        filelist = glob.glob('xiaopiao/prepared/slicestded/*.jpg')[1:]
        #filelist = ['xiaopiao/prepared/slicestded/1_付款：---礼券：5.00，储值-----.jpg']
        #filelist = ['xiaopiao/prepared/slicestded/1_1.请保留好您的购物凭证，凭此单换发票。---.jpg']
    else:
        filelist = [sys.argv[1]]


    infer_main(filelist)

