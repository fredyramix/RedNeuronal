#-*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
import numpy
seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("pares_impares.csv", delimiter=",")
#X = dataset[:,0:2500]
#Y = dataset[:,2500]
X = dataset[:,0:5]
Y = dataset[:,5]


# create model
model = Sequential()
#model.add(Dense(4000, input_dim=2500, init='uniform', activation='relu'))
#model.add(Dense(500, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='sigmoid'))
model.add(Dense(5, input_dim=5, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=5000, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(X)
# round predictions
#rounded = [round(x) for x in predictions]

print(predictions[10])
print(round(predictions[10]))
print ('----------------END---------------------------')