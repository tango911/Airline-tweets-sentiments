# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 23:55:48 2018

@author: MOHIT
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data=pd.read_csv("train.csv")

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

data=data.drop(["tweet_id",
           "airline",
           "name",
           "retweet_count",
           "tweet_created",
           "tweet_location",
           "user_timezone"],axis=1)

#remove words which are starts with @ symbols
data['text'] = data['text'].map(lambda x:re.sub('@\w*','',str(x)))
#remove link starts with https
data['text'] = data['text'].map(lambda x:re.sub('http.*','',str(x)))
#removing data and time (numeric values)
data['text'] = data['text'].map(lambda x:re.sub('[0-9]','',str(x)))
#removing special characters
data['text'] = data['text'].map(lambda x:re.sub('[#|*|$|:|\\|&]','',str(x)))
"""Getting important numeric and non numeric data
Appending negative reason to text
For data['negativereason'] i have removed the NaN values by 0 in 'negativereason' and placed 1 in place of vaild negative reason.
For data['negativereason_confidence'] the values are between 0 to 1 higher the values more its chances to be a 'negative' tweet lower the values more its chances to be 'positive' or 'neutral' tweet. so replacing the NaN by value near to zero
"""
#appending negative reason to text
data=data.values
for i in range(3339):
    if not str(data[i][2])=="nan":
        data[i][4]=str(data[i][4])+" "+ str(data[i][2])
        
#Getting important numeric data 
for i in range(3339):
    if str(data[i][2])=="nan":
        data[i][2]=0
    if str(data[i][3])=="nan":
        data[i][3]=0.3
for i in range(3339):
    if not str(data[i][2])=='0':
        data[i][2]=1
data=pd.DataFrame(data=data,columns=["airline_sentiment",
                                     "airline_sentiment_confidence",
                                     "negativereason",
                                     "negativereason_confidence",
                                     "text"])        
#preparing train data
#removing stopwords and tokenizing it.
stop=stopwords.words('english')
text=[]
none=data['text'].map(lambda x:text.append(' '.join
       ([word for word in str(x).strip().split() if not word in set(stop)])))
tfid=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
x_features=tfid.fit_transform(text).toarray()
#preparing target variable
y=data['airline_sentiment']
y=pd.DataFrame(y,columns=['airline_sentiment'])
y = y['airline_sentiment'].map({'neutral':1,'negative':2,'positive':0})


#emmbading numerical data in x_features
x_features=pd.DataFrame(x_features)
x_features.loc[:,'a']=data.iloc[:,1].values
x_features.loc[:,'b']=data.iloc[:,2].values
x_features.loc[:,'c']=data.iloc[:,3].values

#training by ANN
#lets dig deeper and apply Deep learning for better accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
from sklearn.preprocessing import OneHotEncoder

onehotencoder=OneHotEncoder()
target=y.values
target=target.reshape(-1,1)
target=onehotencoder.fit_transform(target).toarray()
target=pd.DataFrame(data=target,columns=['positive','neutral','negative'])


# ann
clf=Sequential()
#adding layers to ANN
clf.add(Dense(units=2048,activation="relu",kernel_initializer="uniform",kernel_regularizer=regularizers.l2(0.001),input_dim=6212))
clf.add(Dropout(0.5))
#adding two more hidden layer to ANN
clf.add(Dense(units=2048,activation="relu",kernel_initializer="uniform",kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.5))
clf.add(Dense(units=2048,activation="relu",kernel_initializer="uniform",kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.5))
#adding output layer
clf.add(Dense(units=3,activation="softmax",kernel_initializer="uniform"))
#compiling ANN
clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting ANN
hist=clf.fit(x_features,target,batch_size=3,epochs=10)

# Plot the loss and accuracy curves for training and validation 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,1)
ax[0].plot(hist.history['loss'], color='b', label="Training loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(hist.history['acc'], color='r', label="Training accuracy")
legend = ax[1].legend(loc='best', shadow=True)