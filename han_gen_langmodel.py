###### LANGUAGE_MODEL_ADDED #######
#coding=utf-8
import numpy as np
import pandas as pd


DF = pd.read_csv("freq/han_freq.csv",sep=',',encoding='utf-8').iloc[:,:5]
DF.columns = ['id','char','freq','rank','pct']
# Use Quadratic Weight To Prevent Stop Char And Rare Char Generation
FakeFreq = 50-(DF['freq'].map(np.log)-9.5)**2
FakeFreq = (FakeFreq/FakeFreq.sum()).cumsum()

# Use Quadratic Weight To Prevent Stop Char And Rare Char Generation

DictDF = pd.read_csv("freq/dict.txt",header=None,sep=" ",encoding="utf-8")
DictDF.columns = ["word","freq","lac"]
FakeDictFreq = 50-(7.5-DictDF['freq'].map(np.log))**2
FakeDictFreq = (FakeDictFreq/FakeDictFreq.sum()).cumsum()



CHARSET = ''.join([str(i) for i in range(10)])+''.join([chr(i) for i in range(65,91)])+''.join([chr(i) for i in range(97,123)])

def gen_singlechar():
    global CHARSET
    return CHARSET[np.random.randint(0,len(CHARSET))]

def istruealpha(s):
    i = ord(s)
    return True if (i>=97 and i<=122) or (i>=65 and i<=90) else False

"""
def get_label(s):
    outputlabel = []
    for c in s:
        #print c,
        if c.isdigit():
            outputlabel.append(int(c)+1)
        elif istruealpha(c):
            if c.isupper():
                outputlabel.append(ord(c)-54)
            else:
                outputlabel.append(ord(c)-60)
        elif type(c)!=unicode:
            if c == ':':
                outputlabel.append(3563)
            elif c == '.':
                outputlabel.append(3564)
            elif c == ' ':
                outputlabel.append(3562)
            else:
                outputlabel.append(3562)
        else:
            if c == u'：':
                outputlabel.append(3563)
            elif c == u':':
                outputlabel.append(3563)
            elif c == u'.':
                outputlabel.append(3564)
            elif c == u' ':
                outputlabel.append(3562)
            elif len(DF[DF['char']==c])==0:
                #print repr(c)
                outputlabel.append(3562)
            else:
                outputlabel.append(DF[DF['char']==c].index[0]+63)
    
    return outputlabel

"""

d_get_label = {}
d_get_inverse_label = {}
for i in range(10):
    d_get_label[str(i)] = i+1
    d_get_inverse_label[i+1] = unicode(i)
for i in range(65,91):
    d_get_label[chr(i)] = i-54
    d_get_inverse_label[i-54] = unicode(chr(i))
for i in range(97,123):
    d_get_label[chr(i)] = i-60
    d_get_inverse_label[i-60] = unicode(chr(i))
    
d_get_label[':'] = 3563
d_get_label[u'：'] = 3563
d_get_inverse_label[3563] = unicode(':')
d_get_label['.'] = 3564
d_get_inverse_label[3564] = unicode('.')
d_get_label[' '] = 3562
d_get_inverse_label[3562] = unicode(' ')


for i in range(len(DF['char'])):
    d_get_label[DF['char'][i]] = i + 63
    d_get_inverse_label[i+63] = unicode(DF['char'][i])
    
    
def get_label(st):
    global d_get_label
    return [d_get_label[s] if s in d_get_label else 3562 for s in st]

def get_inverse_label(p):
    global d_get_inverse_label
    return [d_get_inverse_label[item] for item in p]

def isLabelHan(i):
    return ( i >= (0 + 63) ) and ( i <= (3498 + 63) )

# Han String Generator, Possibly With Language Model.
def gen_han(length=3, prefix=0, suffix=0, comma=False, from_dict=False):
    
    if from_dict:
        han_part = ''.join([ DictDF['word'][FakeDictFreq[FakeDictFreq>np.random.rand()].index[0]] for i in xrange(length)])
    else:
        han_part = ''.join([ DF['char'][FakeFreq[FakeFreq>np.random.rand()].index[0]] for i in xrange(length)])
    
    if prefix < 0 :
        return None
    elif prefix == 0 :
        prefixstring = ''
    else:
        prefixstring = unicode(''.join([gen_singlechar() for i in range(prefix)])) 
    
    if suffix < 0 :
        return None
    elif suffix == 0 :
        suffixstring = ''
    else:
        suffixstring = unicode(''.join([gen_singlechar() for i in range(suffix)]))
    if comma:
        finalsuffix = u'：:'
        finalstring = finalsuffix[np.random.randint(0,2)]
    else:
        finalstring = ''
        
    output = prefixstring+han_part+suffixstring+finalstring
    return output, get_label(output)


def gen_num( mxm = 1000 ,fl = False):
    if fl:
        s = '%.2f' % (np.random.randint(0,mxm) + np.random.randint(0,100)/100.0)
    else:
        s = unicode( np.random.randint(0,mxm))
    
    return s, get_label(s)

def gen_blank( bk = 3):
    return u' '*bk, [3562 for i in range(bk)]

def gen_fixlen( fixlen = 55, fltlen = 5, 
               hanlenmax = 3, prefixmax= 2, suffixmax = 2,
               log10nummax = 4, bkfix = 3, bkmax = 4):
    n , s, label = 0, u'', []
    while len(s) < fixlen + np.random.randint(-fltlen, fltlen):
        tmp = np.random.randint(0,2)
        if tmp==0:
            from_dict_flag = np.random.rand()>0.1
            news,newl = gen_han(length = 1 + np.random.poisson(hanlenmax),
                         prefix = np.random.poisson(prefixmax),
                         suffix = np.random.poisson(suffixmax),
                         comma =  bool(np.random.randint(0,2)),
                         from_dict = from_dict_flag)

            s += news
            label.extend(newl)
            
            news, newl = gen_blank( bkfix + np.random.poisson(bkmax) )
            s += news
            label.extend(newl)
            

        elif tmp == 1:
            news, newl = gen_num( mxm = 10**np.random.randint(0,log10nummax),
                         fl = bool(np.random.randint(0,2)) )
            
            s += news
            label.extend(newl)
            news, newl = gen_blank( bkfix + np.random.poisson(bkmax) )
            s += news
            label.extend(newl)
            
    finlabel = np.zeros( fixlen + fltlen )
    for idx,labelet in enumerate(label[:(fixlen + fltlen)]):
        finlabel[idx] = labelet
    return s, finlabel