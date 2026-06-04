# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.preprocessing import Imputer
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras as tf


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import lstm

# Importing the dataset
dataset = pd.read_csv('cardio_vascular.csv')
dataset=dataset.dropna(how="any")
print(dataset)
print(" ")
dataset.loc[:, 'cardio'].replace([2, 3, 4], [1, 1, 1], inplace=True)

print(dataset.info())
print(" ")

print(dataset)

#Data Visualization

#age vs cholestrol
m = dataset['Age']
n = dataset['Chol']
plt.figure(figsize=(4,4))
plt.title("Bar plot graph",fontsize=20)
plt.xlabel("Age",fontsize=15)
plt.ylabel("Cholestrol",fontsize=15)
plt.bar(m,n,label="bar plot",color=["orange"],width=0.5)
plt.legend(loc='best')
plt.show()

#age vs chest pain
m = dataset['Age']
n = dataset['Cp']
plt.figure(figsize=(4,4))
plt.title("Bar plot graph",fontsize=20)
plt.xlabel("Age",fontsize=15)
plt.ylabel("Cp",fontsize=15)
plt.bar(m,n,label="bar plot",color=["orange"],width=0.5)
plt.legend(loc='best')
plt.show()



X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 121)#101


print(X_train)
print(" ")

#EXPLORING THE DATASET


#improved LSTM Layers(Incremental order(5 layer architecture))
model = Sequential()
net = model.add(Dense(32,input_dim=13,activation='relu'))
net = model.add(lstm.lstm_layer(net, 64))
net = model.add(lstm.lstm_layer(net, 128))
net = model.add(lstm.lstm_layer(net, 254))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=1000)

model.save('lstm_train.h5')


y_pred = model.predict(X_test)
y_pred = y_pred.round()
print(y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix :")
print(cm)
acc = accuracy_score(y_test, y_pred)
print("Accuracy_score :")
print(acc)
print(" ")

testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes, average='micro')
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes,average='micro')
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes, average='micro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

