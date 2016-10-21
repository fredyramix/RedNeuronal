#-*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import model_from_json
import os
print '========================================================================'
dataset = numpy.loadtxt("DatosEntranamiento/completo20001.txt", delimiter=",")
X = dataset[:,0:2000]
Y = dataset[:,2000]
nuevo = X[0]
json_file = open('model.json','r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
#
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop' ,metrics=['accuracy'])
scores = loaded_model.evaluate(X,Y ,verbose=os)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
print type(X)
print type(X[0])

predictions = loaded_model.predict(numpy.array(X[0]))
# round predictions
rounded = [round(x) for x in predictions]
print rounded