#coding=utf-8

import sys, os, re
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
from io import BytesIO

from han_gen_langmodel import *


#########

def get_padded_label(s,cutoff=1):
    out = np.zeros(cutoff)
    beforepad = get_label(s)
    #if np.all(beforepad>=63) and np.all(beforepad<=3499)
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

##########


class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class OCRIter(mx.io.DataIter):
    def __init__(self , batch_size, num_label, height, width, path ):
        super(OCRIter, self).__init__()
        
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        self.path = path
        
        
    def __iter__(self):
        print 'iter step'

        dir_list = os.listdir(self.path)
        pic_num = len(dir_list)
        num = 0
        
        for k in range(pic_num / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                
                if num > pic_num-1:
                    break

                badcnt = 0
                while 1:
                    img = cv2.imread(self.path + '/' + dir_list[num])
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
                """ 
                lable_value = dir_list[num].split('.')[0]
                lable_value = lable_value.split('_')[1]
                lable_value = re.sub(u'-',u' ',lable_value.decode('utf-8'))
                num += 1
                #img = cv2.resize(img, (900, 30))
                #img = img.transpose(1, 0)
                #img = img.reshape((32 * 32))
                img = img.transpose(2,0,1)
                img = np.float32(np.multiply(img, 1/255.0))

                data.append(img)
                #print len(get_padded_label(lable_value)),
                label.append(get_padded_label(lable_value))
                

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

    
    
def get_ocrnet():
    
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")
    
    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 512)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 3565)
    #fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    #fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    #fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    #fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim = 0)
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.Reshape(data = label, target_shape = (0, ))
    return mx.symbol.SoftmaxOutput(data = fc21, label = label, name = "softmax")


CHARLEN = 1
#CHARLEN = 4

def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / CHARLEN):
        ok = True
        for j in range(CHARLEN):
            k = i * CHARLEN + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total

if __name__ == '__main__':
    
    network = get_ocrnet()
    devs = [mx.gpu(1)]
    
    prefix = "cnn-ocr-fix1-expand-allmax"
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    
    batch_size = 128
    data_train = OCRIter( batch_size, CHARLEN , 32, 32, '/data1/img_data/train_data_1_CHLang_Expand/' )
    data_test = OCRIter( batch_size, CHARLEN , 32, 32, '/data1/img_data/test_data_1_CHLang_Expand/')
    
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    model.fit(X = data_train, eval_data = data_test, eval_metric = Accuracy,
                      batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                      epoch_end_callback = mx.callback.do_checkpoint(prefix, 1))
    
    model.save(prefix)