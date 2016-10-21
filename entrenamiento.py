#-*- coding:utf-8 -*-
#from keras.models import Sequential
#from keras.layers import Dense
import numpy
#from keras.models import model_from_json
import os
def main():
	seed = 7
	numpy.random.seed(seed)
	dataset = numpy.loadtxt("DatosEntranamiento/completo20001.txt", delimiter=",")
	print len(dataset[0])
	numero_muestras = len(dataset)
	tam_vector = len(dataset[0]) -1 
	inputs = dataset[:,0:tam_vector]
	resultados = dataset[:,tam_vector]
	print resultados 
	#X = dataset[:,0:2000]
	#Y = dataset[:,2000]


main()