#-*- coding:utf-8 -*-
'''
Prueba 1


'''
import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
import time


def save(name,model,epo):
	print 'Guardando Datos...'
	model_json = model.to_json()
	nombre_archivo_modelo = 'Entrenamiento/Modelo/'
	nombre_archivo_modelo += name+'_' + str(epo) +'.json'
	with open(nombre_archivo_modelo,'w') as json_file:
		json_file.write(nombre_archivo_modelo)

	name_pesos = 'Entrenamiento/Pesos/'
	name_pesos  += name+'_' + str(epo) +'.h5'
	model.save_weights(name_pesos)
	print 'Guardando Datos Finalizado'
def generarGrafica(name,history,epo):
	nombre_imagen_acc = 'Entrenamiento/Graficas/'
	nombre_imagen_acc +=   name+'_'+ str(epo) +'accuracy.png'

	nombre_imagen_loss = 'Entrenamiento/Graficas/'
	nombre_imagen_loss += name+'_' + str(epo) +'loss.png'

	print 'Creando Graficas'
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(nombre_imagen_acc)
	#plt.show()
	
	#plt.close()
	# summarize history for loss 
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(nombre_imagen_loss)
	#plt.show()
	
	#plt.close()
	print 'Fin creado graficas'
def createModel(capas_ocultas,tam_inputs,datos_compilacion,seed):
	bandera = False
	model = Sequential()
	for i in capas_ocultas:
		if (bandera==False):
			model.add(Dense(i[0]['neuronas'], input_dim=tam_inputs,init=i[1]['init'],activation=i[2]['activation']))
			bandera = True
		else:
			model.add(Dense(i[0]['neuronas'],init=i[1]['init'],activation=i[2]['activation']))
	print 'Compilando...'
	model.compile(loss=datos_compilacion[0]['loss'],optimizer=datos_compilacion[1]['optimizer'],metrics=[datos_compilacion[2]['metrics']])
	return model
def leerEntradas(nombreArchivo,seed):
	dataset = numpy.loadtxt(nombreArchivo, delimiter=",")
	num_muestras = len(dataset)
	tam_vector = len(dataset[0])-1
	print 'Numero de muestras: '+str(num_muestras)
	print 'Tam de Vector: '+str(tam_vector)
	X = dataset[:,0:tam_vector]
	Y = dataset[:,tam_vector]
	return num_muestras,tam_vector,X,Y
def fitModel(model,epocas,batch,X,Y):
	#Fit model
	print 'Entrenando Modelo...'
	history = model.fit(X, Y,nb_epoch=epocas, validation_split=0.33, batch_size=batch,verbose=0)
	print 'Fin Entrenando Modelo'
	return history
def evaluate(X,Y,model):
	#Evaluacion del modelo
	print 'Evaluando Modelo'
	scores = model.evaluate(X, Y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	print 'Fin evaluando Modelo'
def evaluateCrossValidation(model,X,Y,seed):
	kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
	results = cross_val_score(model, X, Y, cv=kfold)
	print(results.mean())
def predecir(X,Y,model):
	#predicciones
	print 'Haciendo Predicciones con Datos de Entrenamiento...'
	predictions = model.predict(X)
	# round predictions
	rounded = [round(x) for x in predictions]
	print 'Reales: '
	print Y
	print 'Predicciones: '
	print rounded

def definirTopologia():
	print 'hola'
	#Numero de Capas.
	#Cantidad de Neuronas por Capa.
	#Grado de Conectividad
	#Tipo de Conexion entre neuronas.
def automaticVerificationDataset(X,Y,model,epocas,batch):
	history=model.fit(X,Y,validation_split=0.33,nb_epoch=epocas,batch_size=batch)
	return history
def manualVerificationDataset(X,Y,model,epocas,batch,seed):
	x_train, x_test, y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=seed)
	history = model.fit(x_train,y_train, validation_data=(x_test,y_test), nb_epoch=epocas,batch_size=batch)
	return history
def manualCrossValidation(X,Y,model,seed,capas_ocultas,tam_vector,datos_compilacion):
	kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
	cvscores = []

	for i, (train, test) in enumerate(kfold):
  	# create model
		model = Sequential()
		model = createModel(capas_ocultas,tam_vector,datos_compilacion,seed)

		history = model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
		# evaluate the model
		scores = model.evaluate(X[test], Y[test], verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)

	print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
	return history

def main():
	seed = 7
	numpy.random.seed(seed)
	pruebas = [150,1000,10000]
	nombre_general = 'Prueba1'
	
	batch_size = 10
	num_muestras,tam_vector,X,Y = leerEntradas('DatosEntranamiento/completo20001.txt',seed)
	tiempos = []
	for i in pruebas:
		num_epocas = i
		#Definicion de Topologia	
		capas_ocultas = [[{'neuronas':1000},{'init':'uniform'},{'activation':'relu'}],[{'neuronas':1},{'init':'uniform'},{'activation':'sigmoid'}]]
		datos_compilacion= [{'loss':'binary_crossentropy'},{'optimizer':'adam'},{'metrics':'accuracy'}]
		#model = KerasClassifier(build_fn=createModel(capas_ocultas,tam_vector,datos_compilacion,seed), nb_epoch=150, batch_size=batch_size, verbose=0)
		modelo = createModel(capas_ocultas,tam_vector,datos_compilacion,seed)
		#evaluateCrossValidation(modelo,X,Y,seed)
		#modelo = Sequential()
		print 'Modelo creado y Compilado'
		start = time.time()
		history = fitModel(modelo,num_epocas,batch_size,X,Y)
		end = time.time()
		#print 'Tiempo de Entrenamiento: '
		final= end-start
		#history = automaticVerificationDataset(X,Y,modelo,num_epocas,batch_size)
		#history = manualVerificationDataset(X,Y,modelo,num_epocas,batch_size,seed)
		#history = manualCrossValidation(X,Y,modelo,seed,capas_ocultas,tam_vector,datos_compilacion)
		evaluate(X,Y,modelo)
		predecir(X,Y,modelo)
		save(nombre_general,modelo,num_epocas)
		generarGrafica(nombre_general,history,num_epocas)
		print nombre_general + ": Finalizado con " + str(i) +" epocas"
		tiempos.append(final)
	print "Tiempos de Entrenamiento: "
	print tiempos
main()