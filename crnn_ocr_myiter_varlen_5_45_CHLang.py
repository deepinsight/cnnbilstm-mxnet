#coding=utf-8
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from cnn_lstm_model import crnn_unroll

from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random
import os
import re

from han_gen_langmodel import *

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


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
    

class OCRIter(mx.io.DataIter):
    def __init__(self, batch_size, num_label, init_states, path, check):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.num_label = num_label
        self.init_states = init_states
        self.path = path
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 1, 32, 900))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.check = check

    def __iter__(self):
        print 'iter'
        init_state_names = [x[0] for x in self.init_states]
        dir_list = os.listdir(self.path)
        pic_num = len(dir_list)
        num = 0
        for k in range(pic_num / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                if num > pic_num-1:
                    break
                """
                badcnt = 0
                while 1:
                    img = cv2.imread(self.path + '/' + dir_list[num], 0)
                    if img != None or badcnt > 10:
                        break
                    badcnt += 1
                if badcnt > 10:
                    print dir_list[num]
                """
                while 1:
                    img = cv2.imread(self.path + '/' + dir_list[num], 0)
                    if img != None:
                        break
                        
                lable_value = dir_list[num][:-4]
                lable_value = lable_value.split('_')[1]
                lable_value = re.sub(u'-',u' ',lable_value.decode('utf-8'))
                num += 1
                #img = cv2.resize(img, (900, 30))
                #img = img.transpose(1, 0)
                newimg = np.zeros((32 , 900))+255.0
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        newimg[i+1,j] = img[i,j]
                newimg = newimg.reshape((1 , 32 , 900))
                newimg = np.float32(np.multiply(newimg, 1/255.0))

                data.append(newimg)
                #print len(get_padded_label(lable_value)),
                label.append(get_padded_label(lable_value))

            #print(label,type(label))
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass



BATCH_SIZE = 128
SEQ_LENGTH = 55

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
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


def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = label[i]
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

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

def LCS_Purge(p,l):
    mp = [item for item in p if item!=3562]
    ml = [item for item in l if item!=3562]
    if len(mp)==0 or len(ml)==0:
        return 0
    P = np.array(list(mp)).reshape((1,len(mp)))
    L = np.array(list(ml)).reshape((len(ml),1))
    M = np.int32(P==L)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            up = 0 if i==0 else M[i-1,j]
            left = 0 if j==0 else M[i,j-1]
            M[i,j] = max(up,left, M[i,j] if (i==0 or j==0) else M[i,j]+M[i-1,j-1])
    return M.max()


def Accuracy_LCS(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    global CTX_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        
        ## Dynamic Programming Finding LCS
        if len(l)==0:
            continue
        hit += LCS(p,l)*1.0/len(l)
        total += 1.0
    return hit / total

def Accuracy_LCS_Purge(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    global CTX_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        
        ## Dynamic Programming Finding LCS
        ml = [item for item in l if item!=3562]
        ml = remove_trailing_zero(ml)
        if len(ml)==0:
            pass
        else:
            hit += LCS_Purge(p,ml)*1.0/len(ml)
            total += 1.0
    if total == 0:
        print "Total:0"
        return 0
    else:
        return hit / total

def isLabelHan(i):
    return ( i >= (0 + 63) ) and ( i <= (3498 + 63) )


def Accuracy_LCS_Purge_HanNon(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    global CTX_LENGTH
    hit = 0.
    total = 0.
    
    hithan = 0.
    totalhan = 0.
    hitnon = 0.
    totalnon = 0.
    
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        
        ## Dynamic Programming Finding LCS
        ml = [item for item in l if item!=3562]
        ml = remove_trailing_zero(ml)
        if len(ml)==0:
            pass
        else:
            hit += LCS_Purge(p,ml)*1.0/len(ml)
            total += 1.0
            
        ## Han LCS
        phan = [item for item in p if isLabelHan(item)]
        mlhan = [item for item in ml if isLabelHan(item)]
        
        if len(mlhan)==0:
            pass
        else:
            hithan += LCS_Purge(phan, mlhan)*1.0/len(mlhan)
            totalhan += 1.0
        
        ## Non-Han LCS
        pnon = [item for item in p if not isLabelHan(item)]
        mlnon = [item for item in ml if not isLabelHan(item)]
        
        if len(mlnon)==0:
            pass
        else:
            hitnon += LCS_Purge(pnon, mlnon)*1.0/len(mlnon)
            totalnon += 1.0
        
        #print ml
        #print mlhan
        #print mlnon
        #break
        
    if totalhan==0:
        print "totalhan:0"
        #return 0
    else:
        return hithan / totalhan if totalhan > 0.0 else 0 
        #return hitnon / totalnon if totalnon > 0.0 else 0
        #return  hit / total
        #                 (hithan / totalhan if totalhan > 0.0 else 0 ),
        #                 (hitnon / totalnon if totalnon > 0.0 else 0 )]
                         


    

if __name__ == '__main__':
    
    num_hidden = 1024
    num_lstm_layer = 2
    num_epoch = 100
    learning_rate = 0.0003
    momentum = 0.9
    num_label = 45

    # 256*2, lr = 0.002 Best Ever
    
    prefix = 'mymodel/CRNN_BN_512_1024_1024_1024_ADADELTAMIXLANGUAGE_BATCH{bat}_SEQ{seq}_varlen_{hid}-{nlay}-{lr}-100-mmt{mmt}-predhan'.\
            format(bat=BATCH_SIZE,seq=SEQ_LENGTH,hid=num_hidden,nlay=num_lstm_layer,lr=learning_rate,mmt=momentum)
    iteration = 10
    contexts = [mx.context.gpu(7)]
    
    print prefix

    def sym_gen(seq_len):
        return crnn_unroll(num_lstm_layer, seq_len,
                           num_hidden= num_hidden,
                           num_label = num_label)

    init_c = [('l%d_init_c'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    train_path  = '/data2/img_data/train_data_varlen_5_45_CHLang_ConcatMix/'
    test_path  = '/data2/img_data/test_data_varlen_5_45_CHLang_ConcatMix/'

    data_train = OCRIter(BATCH_SIZE, num_label, init_states, train_path, 'train')
    data_val = OCRIter(BATCH_SIZE, num_label, init_states, test_path, 'test')

    symbol = sym_gen(SEQ_LENGTH)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 #momentum=momentum,
                                 optimizer='adadelta',
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    # load model
    # model = mx.model.FeedForward.load(prefix,iteration,
    #                                 learning_rate = learning_rate,
    #                                 ctx = contexts,
    #                                 numpy_batch_size = BATCH_SIZE,
    #                                 num_epoch=num_epoch)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    
    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy_LCS_Purge_HanNon),
              batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 100),
              epoch_end_callback = mx.callback.do_checkpoint(prefix, 1))

    model.save(prefix, iteration)
