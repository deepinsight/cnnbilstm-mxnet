# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len,
                num_hidden, num_label):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=3565)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)
    return sm


def lstm_inference_symbol(num_lstm_layer, seq_len, num_hidden, num_label):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=3565)
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)


###################################################

def crnn_unroll(num_lstm_layer, seq_len ,
                num_hidden, num_label):

    
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    
    #wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)
    
    # CNN PART:
    # Conv(3,3):64->MaxPool(2,2):2->Relu->Conv(3,3):128->MaxPool(2,2):2->Relu->
    # Conv(3,3):256->Relu->Conv(3,3):256->MaxPool(1,2):2->Relu->
    # Conv(3,3):512->MaxPool(1,2):2->Conv(2,2):512
    conv1 = mx.symbol.Convolution(data=data, kernel=(3,3), pad=(1, 1) , stride=(1, 1), num_filter=64)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), pad=(1, 1), stride=(1, 1), num_filter=128)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3,3), pad=(1, 1),stride=(1, 1), num_filter=256)
    #relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    
    conv4 = mx.symbol.Convolution(data=conv3, kernel=(3,3), pad=(1, 1), stride=(1, 1), num_filter= 512)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="max", kernel= (1 , 2), stride = (2, 2) )
    #relu4 = mx.symbol.Activation(data=pool4, act_type="relu")
    
    bn1 = mx.symbol.BatchNorm(data=pool4)
    conv5 = mx.symbol.Convolution(data=bn1, kernel=(3,3), pad=(1, 1), stride=(1, 1), num_filter = 1024 )
    bn2 = mx.symbol.BatchNorm(data=conv5)
    
    pool5 = mx.symbol.Pooling(data=bn2, pool_type="max", kernel=(1, 2), stride = (2, 2))
    conv6 = mx.symbol.Convolution(data=pool5, kernel = (2,2), stride=(1, 1),  num_filter = 1024 )
    
    # 900 after 4 pooling: 55
    trans = mx.sym.transpose(data=conv6,axes=(0,3,1,2))
    flatten = mx.sym.Flatten(data=trans)
    wordvec = mx.sym.SliceChannel(data=flatten, num_outputs=seq_len, squeeze_axis=1)
    
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=3565, weight=cls_weight, bias=cls_bias, name='pred')

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)
    
    return sm


def crnn_inference_model(num_lstm_layer, seq_len ,
                num_hidden, num_label):

    
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    
    #wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)
    
    # CNN PART:
    # Conv(3,3):64->MaxPool(2,2):2->Relu->Conv(3,3):128->MaxPool(2,2):2->Relu->
    # Conv(3,3):256->Relu->Conv(3,3):256->MaxPool(1,2):2->Relu->
    # Conv(3,3):512->MaxPool(1,2):2->Conv(2,2):512
    conv1 = mx.symbol.Convolution(data=data, kernel=(3,3), pad=(1, 1) , stride=(1, 1), num_filter=64)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), pad=(1, 1), stride=(1, 1), num_filter=128)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    #relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3,3), pad=(1, 1),stride=(1, 1), num_filter=256)
    #relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    
    conv4 = mx.symbol.Convolution(data=conv3, kernel=(3,3), pad=(1, 1), stride=(1, 1), num_filter= 512)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="max", kernel= (1 , 2), stride = (2, 2) )
    #relu4 = mx.symbol.Activation(data=pool4, act_type="relu")
    
    bn1 = mx.symbol.BatchNorm(data=pool4)
    conv5 = mx.symbol.Convolution(data=bn1, kernel=(3,3), pad=(1, 1), stride=(1, 1), num_filter = 1024 )
    bn2 = mx.symbol.BatchNorm(data=conv5)
    
    pool5 = mx.symbol.Pooling(data=bn2, pool_type="max", kernel=(1, 2), stride = (2, 2))
    conv6 = mx.symbol.Convolution(data=pool5, kernel = (2,2), stride=(1, 1),  num_filter = 1024 )
    
    # 900 after 4 pooling: 57
    wordvec = mx.sym.SliceChannel(data=conv6, num_outputs=seq_len, squeeze_axis=1)
    
    seqidx
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=3565, weight=cls_weight, bias=cls_bias, name='pred' )

    sm = mx.sym.SoftmaxOutput(data=pred, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
