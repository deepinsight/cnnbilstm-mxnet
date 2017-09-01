# coding=utf-8
import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append("/home/liang/frankwang/mxnet/amalgamation/python/")
sys.path.append("/home/liang/frankwang/mxnet/python/")

from lstm import *
import mxnet as mx
import numpy as np
import cv2
import os


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
        self.provide_data = [('data', (batch_size, 900 * 30))] + init_states
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
                        
                lable_value = dir_list[num].split('.')[0]
                lable_value = lable_value.split('_')[1]
                lable_value = re.sub(u'-',u' ',lable_value.decode('utf-8'))
                num += 1
                #img = cv2.resize(img, (900, 30))
                img = img.transpose(1, 0)
                img = img.reshape((900 * 30))
                img = np.float32(np.multiply(img, 1/255.0))

                data.append(img)
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
    

'''
# mymodel/LANGUAGE_BATCH128_SEQ60_varlen_1024-3-0.0005-100-mmt0.8-predhan-0037.params

class lstm_ocr_model(object):
    # Keep Zero index for blank. (CTC request it)
    # CONST_CHAR='0123456789'
    def __init__(self, path_of_json, path_of_params):
        super(lstm_ocr_model, self).__init__()
        self.path_of_json = path_of_json
        self.path_of_params = path_of_params
        self.predictor = None
        self.__init_ocr()

    def __init_ocr(self):
        num_label = 45 # Set your max length of label, add one more for blank
        batch_size = 1

        num_hidden = 1024
        num_lstm_layer = 3
        init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_states = init_c + init_h

        all_shapes = [('data', (batch_size, 80 * 30))] + init_states + [('label', (batch_size, num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]
        self.predictor = Predictor(open(self.path_of_json).read(),
                                    open(self.path_of_params).read(),
                                    all_shapes_dict)

    def forward_ocr(self, img):
        img = cv2.resize(img, (80, 30))
        img = img.transpose(1, 0)
        img = img.reshape((80 * 30))
        img = np.multiply(img, 1/255.0)
        self.predictor.forward(data=img)
        prob = self.predictor.get_output(0)
        label_list = []
        for p in prob:
            max_index = np.argsort(p)[::-1][0]
            label_list.append(max_index)
        return self.__get_string(label_list)

    def __get_string(self, label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        # change to ascii
        s = ''
        for l in ret:
            if l > 0 and l < (len(lstm_ocr_model.CONST_CHAR)+1):
                c = lstm_ocr_model.CONST_CHAR[l-1]
            else:
                c = ''
            s += c
        return s
'''


if __name__ == '__main__':
    
    prefix = 'mymodel/LANGUAGE_BATCH128_SEQ60_varlen_1024-3-0.0005-100-mmt0.8-predhan'
    iteration = 37

    model = mx.model.FeedForward.load(prefix,iteration)

    batch_size = 1
    num_label = 45
    num_hidden = 1024
    num_lstm_layer = 3
    SEQ_LENGTH = 60
    
    num_epoch = 100
    learning_rate = 0.0005
    momentum = 0.8
    
    contexts = [mx.context.gpu(7)]
    
    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                           num_hidden= num_hidden,
                           num_label = num_label)
    
    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    
    symbol = sym_gen(SEQ_LENGTH)

    data_input = [("data", (batch_size, ))] 
    
    input_shapes = dict(init_c + init_h + data_input)
    executor = symbol.simple_bind(ctx=[mx.context.gpu(2)], **input_shapes)

    test_path  = './xiaopiao/prepared/slicestded/'

    data_val = OCRIter(BATCH_SIZE, num_label, init_states, test_path, 'test')
    
    model.predict(data_val,merge_batches=False)
    
        
