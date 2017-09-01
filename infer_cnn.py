#coding=utf-8
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
from io import BytesIO

from han_gen_langmodel import *

def get_ocrnet():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")
    
    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 512)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 3565)
    #fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    #fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    #fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    #fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim = 0)
    #return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")
    return mx.symbol.SoftmaxOutput(data = fc21, name = "softmax")

if __name__ == '__main__':
    
    THRESHOLD = 10.0/255.0
    MAXWIDTH = 32
    MAXHEIGHT = 32
    
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("cnn-ocr-fix1", 100)
    data_shape = [("data", (batch_size, 3, 32, 32))]
    input_shapes = dict(data_shape)
    
    if len(sys.argv)==1:
        #filelist = glob.glob('xiaopiao/prepared/slicestded/*.jpg')
        filelist = ['200000_9建三江-.jpg']
    else:
        filelist = [sys.argv[1]]
        
    for filename in filelist:
        img = cv2.imread(filename)
        img = img.transpose((2,0,1))
        hist0 = np.mean((img[0,:,:] - 255.0)*(-1/255.0),axis=0)
        img = img * 1.0 / 255.0
        start , end = 0,0
                
        res = []

        for i in range(1,len(hist0)):
            if hist0[i]<=THRESHOLD or i==(len(hist0)-1):
                end = i
                if end-start<=3:
                    start,end = i,i
                    continue
                else:
                    part_img = img[:, :, start:end ]
                    print start,end
                    clean_img = np.concatenate([cv2.resize(part_img[k,:,:],(32,32)).transpose(0,1).reshape((1,32,32)) for k in range(3) ],axis=0)
                    clean_img = np.float32(clean_img)
                    #print img.shape, part_img.shape
                    print clean_img.transpose((1,2,0)).shape
                    cv2.imwrite('test'+str(i)+'.jpg',(clean_img*255).transpose((1,2,0)))
                    sym = get_ocrnet()
                    executor = sym.simple_bind(ctx = mx.gpu(7), **input_shapes)
                    for key in executor.arg_dict.keys():
                        if key in arg_params:
                            arg_params[key].copyto(executor.arg_dict[key])

                    executor.forward(is_train = False, data = mx.nd.array([clean_img]))
    
                    probs = executor.outputs[0].asnumpy()
        
                    #probs = executor.forward(mx.nd.array(img)).outputs[0].asnumpy()
                    print probs.shape
                    for k in range(probs.shape[0]):
                        tmpprobs = probs[k]
                        print max(tmpprobs),min(tmpprobs)
                        print np.argmax(tmpprobs), tmpprobs[np.argmax(tmpprobs)]
                        res.append(np.argmax(probs[k]))
                    
                    start,end = i,i
                    
            else:
                end = i
                continue
                
        #p = ctc_label(res)
        print 'Predicted label: ' + str(res)
        print 'Predicted ocr: ' + ''.join(get_inverse_label(res))
        