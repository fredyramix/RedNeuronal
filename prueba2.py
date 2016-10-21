#-*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import model_from_json
import os
seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("DatosEntranamiento/completo20001.txt", delimiter=",")
X = dataset[:,0:2000]
Y = dataset[:,2000]
print 'El numero de muestras es: '+str(len(X))
print 'Datos de Entrenamiento: '


#a=raw_input("Presiona una tecla para continuar")

# create model
print 'Creando modelo...'
model = Sequential()
model.add(Dense(1000, input_dim=2000, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
print 'Compilando Modelo...'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
print 'Entrenar Modelo...'
model.fit(X, Y,validation_split=0.33, nb_epoch=1000, batch_size=10)
# evaluate the model
print '=================Evaluacion de Modelo====='
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
print 'Escribiendo JSON de Modelo y Pesos...'
model_json = model.to_json()
with open('model.json','w') as json_file:
	json_file.write(model_json)
print 'Serializando pesos en H5'
model.save_weights("model.h5")
print 'Modelo guardado'
print 'Predicciones...'
predictions = model.predict(X)
# round predictions
rounded = [round(x) for x in predictions]
print '----------------END TRAINING ---------------------------'
print 'El numero de muestras es: '+str(len(X))
print 'Datos de Entrenamiento: '
print Y
print 'Predicciones de los mismos datos de Entrenamiento: '
print rounded