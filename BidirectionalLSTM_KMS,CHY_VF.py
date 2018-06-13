# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:15:24 2018

@author: HyunYul.Choi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pickle
import os, sys
import nltk
import re
import pandas as pd
import keras
import keras.backend as K
import csv
import numpy as np
import h5py
import re
import nltk
nltk.download('stopwords')
import konlpy.jvm
from konlpy.tag import Twitter; t = Twitter()
from konlpy.corpus import kobill
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from konlpy.corpus import kobill
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
#from drnn import drnn_classification
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

#konlpy.jvm.init_jvm()
java_path = "C:/Program Files/Java/jdk1.8.0_171/bin/java.exe"
os.environ['JAVAHOME'] = java_path
nltk.internals.config_java(options='-Xmx14336m')

#Set GPU memory usage threshold
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

files_ko = kobill.fileids()
os.chdir('C:/ProgramData/Anaconda3/Lib/site-packages/konlpy/data/corpus/kobill')
doc_ko = kobill.open('results.csv').read()

# read corpus.pkl
with open('D:/Dropbox/DMQA_최현율/★Text mining/corpus.pickle', 'rb') as f:
        corpus = pickle.load(f)

AA=len(doc_ko)

# doc=re.sub('\n',',',doc_ko)
doc=doc_ko[0:AA]
doc=re.sub('",','%&*',doc)
doc=re.sub('\n','%&*',doc)
print(doc[1:10])

docs=np.array(doc.split('%&*'))
BB=len(docs)
print(BB)
docs=np.reshape(docs[6:103848],(17307,6))
print(docs[17305:17307])

import datetime
date=docs[:,2]
print(date)

standard_date=datetime.date(2007,3,6)
int_date=[]
for i in range(len(date)):
# for i in range(3):

    dd=date[i].replace('"','')
    yr=int(dd[0:5])
    month=int(dd[6:8])
    day=int(dd[9:11])
    
    d=datetime.date(yr,month,day)
    diff=d-standard_date
    int_date.append(int(diff.days))
# date=re.sub('20','',date)
print(int_date[0:10])

from konlpy.tag import Kkma 
kkma = Kkma()

def clean_text(aa):
    yy= re.sub('[^ㄱ-ㅎ가-힣]', ' ', aa)
    return yy

# Cleaning : 글자 단어, 접사, 조사, 어미, 부호, 모르는 단어 제외
def selector(sentence):
    sentence=clean_text(sentence)
    words=kkma.pos(sentence)
    selected=[]    
    for i in range(np.shape(words)[0]):
        if 'J' in words[i][1]:
            continue
        elif 'E' in words[i][1]:
            continue
        elif 'S' in words[i][1]:
            continue
        elif 'X' in words[i][1]:
            continue
        elif 'U' in words[i][1]:
            continue
        elif len(words[i][0])==1:
            continue
        else:
            selected.append(words[i][0])
            
    return(selected)

selector(docs[0,5])

#stopwords 지정
stop=['트위터','페이스','출처','조인']

#corpus를 문장이 아닌 단어의 나열 형태로 변환합니다. corpus2로 저장했습니다.
for i in range(np.shape(corpus)[0]):
    corpus[i] = [e for e in corpus[i] if e not in stop]
# print(corpus[0:2])
print(np.shape(corpus))

from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', ['words', 'tags'])

docs[:,2]=int_date

tagged_train_docs=[]
for i in range(max(int_date)):
    tagged_train_docs.append(TaggedDocument(corpus[i], [int_date[i]]))
    
print(tagged_train_docs[0:4])

from gensim.models import doc2vec
import numpy as np

vector_size=128
n_day=np.shape(tagged_train_docs)[0]

# 사전 구축 vector size가 차원 수입니다.
#dm
doc_vectorizer_dm = doc2vec.Doc2Vec(vector_size=vector_size,dm_concat=1,dm=1,min_count=10)
doc_vectorizer_dm.build_vocab(tagged_train_docs)


for epoch in range(300):
    doc_vectorizer_dm.train(tagged_train_docs,total_examples=doc_vectorizer_dm.corpus_count,epochs=1)
    doc_vectorizer_dm.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer_dm.min_alpha = doc_vectorizer_dm.alpha  # fix the learning rate, no decay
    if (epoch+1) % 30 ==0:
        print(epoch+1, "epoch 진행중")
print('DM_finished')

# dbow
doc_vectorizer_dbow = doc2vec.Doc2Vec(vector_size=vector_size,dm_concat=1,dm=0,min_count=10)
doc_vectorizer_dbow.build_vocab(tagged_train_docs)


for epoch in range(300):
    doc_vectorizer_dbow.train(tagged_train_docs,total_examples=doc_vectorizer_dbow.corpus_count,epochs=1)
    doc_vectorizer_dbow.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer_dbow.min_alpha = doc_vectorizer_dbow.alpha  # fix the learning rate, no decay
    if (epoch+1) % 30 ==0:
        print(epoch+1, "epoch 진행중")
print('DBOW_finished')

# dm과 dbow를 concat 하여 day matrix를 만듭니다. 이후 전체 day matrix를 일정 길이로 자른 행렬이 Text CNN의 input이 됩니다.
day_matrix=np.zeros((n_day,2*vector_size))
for i in range(n_day):
    day_matrix[i,:]=np.concatenate([doc_vectorizer_dm[i],doc_vectorizer_dbow[i]])

    
#100개의 기사를 2개씩 50일에 해당하는 tag를 붙였으니 embed된 day matrix는 row=50=day의 개수, col=60= dm_vector_30+dbow_vector30
#이 됩니다.
print(day_matrix.shape)
print(day_matrix[0:5])

os.chdir('C:/users/choih/textmining')
price_index=pd.read_csv('APT_price_index.csv',encoding='cp949')
print(price_index.shape)
price_index.head(5)

#총 분석 기간 (134개월)
n_month=price_index.shape[0]
#몇개월 후를 예측할 것인가?
term_month=6


# seoul_index=price_index.loc[:,'서울특별시']
def month_onehot(location,thresh_up=2.5,thresh_down=1.3,term_month=6):
    seoul_index=price_index.loc[:,location]
    updown_onehot=[]
    updown=[]
    count_up=0
    count_down=0
    count_same=0
    for i in range(len(seoul_index)-term_month):
        if seoul_index[i+term_month]>seoul_index[i]+thresh_up:
            updown=[1,0,0]
            updown_onehot.append(updown)
            count_up=count_up+1
        elif seoul_index[i+term_month]<seoul_index[i]+thresh_down:
            updown=[0,0,1]
            updown_onehot.append(updown)
            count_down=count_down+1
        else:
            updown=[0,1,0]
            updown_onehot.append(updown)
            count_same=count_same+1
    print("상승 :",count_up,"보합",count_same,"하락",count_down)
    return(updown_onehot)
            
#상승,하락,보합 비율이 그나마 비슷하게 만든 threshold 값
seoul_month_label=month_onehot('전국',thresh_up=2.5,thresh_down=1.3,term_month=term_month)
print(seoul_month_label)
          
#귀찮아서 한달은 30.4375일의 반올림값으로 정의했습니다. (1년을 365.25일이라고 했을 때의 평균 한달값)
fake_month=30.4375*np.array(range(0,n_month))
for i in range(len(fake_month)):
    fake_month[i]=int(round(fake_month[i]))
    
#가정: 같은 월의 다른 날짜의 label은 같음. -> 3월 1일부터 3월 31일까지의 상승/하락 지표 = 9월 index
print(n_month-term_month)

seoul_day_label=[]
for i in range(n_month-term_month):
    day_diff=int(fake_month[i+1]-fake_month[i])
    replicate=[]
    for j in range(day_diff):
        replicate.append(seoul_month_label[i])
    for k in range(np.shape(replicate)[0]):
        seoul_day_label.append(replicate[k])
        
seoul_day_label=np.array(seoul_day_label)
y=seoul_day_label   

print(seoul_day_label)
print(np.shape(seoul_day_label))   # 이 부분에서 Shape 3 인걸 100으로 바꿔야 함.

# Flow Matrix
n_flow=30
n_matrix_row=n_day-n_flow

y=seoul_day_label
y.shape

# 해당일로부터 (day) 과거 특정 시점 (n_flow)까지의 day vector를 이어붙인 flow matirx를 생성해주는 함수입니다. 
def flow_generator(day,n_flow):
    end=day
    start=end-n_flow
    day_n=day_matrix[start:end,:]
    return(day_n)


# 모든 날에 대해서 30일치 벡터 flow를 만들어 줍니다. 총 4043일에 해당하는 flow matrix가 생성됩니다.
flow_matrix=[]
for i in range(n_matrix_row):
    flow_matrix.append(flow_generator(i+n_flow,n_flow))

flow_matrix=np.array(flow_matrix)
print(np.shape(flow_matrix))

with open('D:/Dropbox/DMQA_최현율/★Text mining/flow_matrix.pickle', 'rb') as f:
        flow_matrix = pickle.load(f)


from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
 
# train test set 구성
#6개월 뒤의 가격이 나온 데이터만 의미가 있으므로 X의 뒤쪽부분 (3896~4043)은 버림
X_train=flow_matrix[0:3000,:]
X_test=flow_matrix[3000:3896,:]
Y_train=y[0:3000,:]
Y_test=y[3000:3896,:]
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

#train_shape=X_train.shape
#test_shape=X_test.shape

#본래 CNN이 이미지데이터를 대상으로 만들었기에 shape가 [index,가로,세로,rgb채널]의 형태로 되어 있습니다.
#이 형태로 맞추기 위해 reshape를 해줍니다.
#X_train=np.reshape(X_train,(train_shape[0],train_shape[1],train_shape[2],1))
#X_test=np.reshape(X_test,(test_shape[0],test_shape[1],test_shape[2],1))
#print(X_test.shape)

# LSTM Shape
#X_train_LSTM=day_matrix[0:3000,:]
#X_test_LSTM=day_matrix[3000:3896,:]
#X_test_LSTM.shape
#
Y_train_LSTM=np.array(Y_train)
Y_test_LSTM=np.array(Y_test)
Y_train_LSTM=np.reshape(Y_train,(3000,1,3))
Y_test_LSTM=np.reshape(Y_test,(896,1,3))
Y_test_LSTM.shape

print(Y_train.shape)
print(Y_test.shape)
print(X_train.shape)
print(X_test.shape)



# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X_train = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	Y_train = array([0 if x < limit else 1 for x in cumsum(X_train)])
	# reshape input and output data to be suitable for LSTMs
	X_train = X_train.reshape(1, n_timesteps, 1)
	Y_train = Y_train.reshape(1, n_timesteps, 1)
	return X_train, Y_train

print(len(y)) 
# That means that instead of the TimeDistributed layer receiving 10 timesteps of 20 outputs,
# it will now receive 10 timesteps of 40 (20 units + 20 units) outputs.

# define problem properties
n_timesteps = 10  #3000 
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))  # 3000,30
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X_train,Y_train = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2)
#   model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2, validation_split=0.1)

# evaluate LSTM
X_train,Y_train = get_sequence(n_timesteps)
Yhat = model.predict_classes(X_train, verbose=0)
for i in range(n_timesteps):
	print('Expected:', Y_train[0, i], 'Predicted', Yhat[0, i])
    
model.evaluate(X_train, Y_train)
 
# Save the model
filepath = 'D:/Dropbox/DMQA_최현율/★Text mining/BidirectionalLSTM_VF.h5'
model.save(filepath)
