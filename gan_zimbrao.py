# original reference: https://github.com/Zackory/Keras-MNIST-GAN

import sys
import os
import shutil
from keras.datasets import mnist
import numpy as np
import math
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Sequential
from keras.layers import Reshape, Dense, Dropout, Flatten, Conv2D, LeakyReLU, Activation, Input, concatenate
import matplotlib.pyplot as plt
import matplotlib.finance
import time
import pandas as pd
from sklearn.metrics import mean_squared_error

np.random.seed(7)


train = pd.read_csv('minidolar/train.csv', sep = ',',  engine='python', decimal='.',header=0)
test = pd.read_csv('minidolar/test.csv', sep = ',',  engine='python', decimal='.',header=0)

train_shift = train['shift']
train_target = train['f0']
train_open = train[['v0','v4','v8','v12','v16','v20','v24','v28','v32','v36','v40','v44','v48','v52','v56','v60','v64','v68','v72','v76','v80','v84','v88','v92','v96','v100','v104','v108','v112','v116']]
train_high = train[['v1','v5','v9','v13','v17','v21','v25','v29','v33','v37','v41','v45','v49','v53','v57','v61','v65','v69','v73','v77','v81','v85','v89','v93','v97','v101','v105','v109','v113','v117']]
train_low = train[['v2','v6','v10','v14','v18','v22','v26','v30','v34','v38','v42','v46','v50','v54','v58','v62','v66','v70','v74','v78','v82','v86','v90','v94','v98','v102','v106','v110','v114','v118']]
train_close = train[['v3','v7','v11','v15','v19','v23','v27','v31','v35','v39','v43','v47','v51','v55','v59','v63','v67','v71','v75','v79','v83','v87','v91','v95','v99','v103','v107','v111','v115','v119']]

test_shift = test['shift']
test_target = test['f0']
test_open = test[['v0','v4','v8','v12','v16','v20','v24','v28','v32','v36','v40','v44','v48','v52','v56','v60','v64','v68','v72','v76','v80','v84','v88','v92','v96','v100','v104','v108','v112','v116']]
test_high = test[['v1','v5','v9','v13','v17','v21','v25','v29','v33','v37','v41','v45','v49','v53','v57','v61','v65','v69','v73','v77','v81','v85','v89','v93','v97','v101','v105','v109','v113','v117']]
test_low = test[['v2','v6','v10','v14','v18','v22','v26','v30','v34','v38','v42','v46','v50','v54','v58','v62','v66','v70','v74','v78','v82','v86','v90','v94','v98','v102','v106','v110','v114','v118']]
test_close = test[['v3','v7','v11','v15','v19','v23','v27','v31','v35','v39','v43','v47','v51','v55','v59','v63','v67','v71','v75','v79','v83','v87','v91','v95','v99','v103','v107','v111','v115','v119']]


# obtendo train target para O,H,L
train_l_target = []
train_h_target = []
train_o_target = []
for i in range(train_shift.shape[0]-1):
    train_l_target.append(np.array(train_low)[i+1][29]+train_shift[i+1]-train_shift[i])
    train_h_target.append(np.array(train_high)[i+1][29]+train_shift[i+1]-train_shift[i])
    train_o_target.append(np.array(train_open)[i+1][29]+train_shift[i+1]-train_shift[i])

    
train_ohlc_target = np.column_stack((train_o_target, train_h_target, train_l_target, train_target[:-1]))

# obtendo test target para O,H,L
test_l_target = []
test_h_target = []
test_o_target = []
for i in range(test_shift.shape[0]-1):
    test_l_target.append(np.array(test_low)[i+1][29]+test_shift[i+1]-test_shift[i])
    test_h_target.append(np.array(test_high)[i+1][29]+test_shift[i+1]-test_shift[i])
    test_o_target.append(np.array(test_open)[i+1][29]+test_shift[i+1]-test_shift[i])

    
test_ohlc_target = np.column_stack((test_o_target, test_h_target, test_l_target, test_target[:-1]))

#removendo ultimo elemento pq é perdido ao calcular os targets O,H e L
train_shift = train_shift[:-1]
train_target = train_target[:-1]
train_open = train_open[:-1]
train_high = train_high[:-1]
train_low = train_low[:-1]
train_close = train_close[:-1]

test_shift = test_shift[:-1]
test_target = test_target[:-1]
test_open = test_open[:-1]
test_high = test_high[:-1]
test_low = test_low[:-1]
test_close = test_close[:-1]


X_train, X_test, Y_train, Y_test =  np.column_stack((train_open.values,train_high.values,train_low.values,train_close.values)),  np.column_stack((test_open.values,test_high.values,test_low.values,test_close.values)), train_ohlc_target, test_ohlc_target #target OHLC
#X_train, X_test, Y_train, Y_test =  np.column_stack((train_open.values,train_high.values,train_low.values,train_close.values)),  np.column_stack((test_open.values,test_high.values,test_low.values,test_close.values)),  np.array(train_target.values.reshape(train_target.size,1)),  np.array(test_target.values.reshape(test_target.size,1))
X_trainp, X_testp, Y_trainp, Y_testp = X_train+train_shift.values.reshape(train_shift.size,1), X_test+test_shift.values.reshape(test_shift.size,1), Y_train+train_shift.values.reshape(train_shift.size,1), Y_test + test_shift.values.reshape(test_shift.size,1)
X_train = np.column_stack((X_train,Y_train)) # fica 120 + colunas target(1 ou 4) "features"


#apenas close
# X_train, X_test, Y_train, Y_test =  np.array(train_close),  np.array(test_close),  np.array(train_target.values.reshape(train_target.size,1)),  np.array(test_target.values.reshape(test_target.size,1))
# X_trainp, X_testp, Y_trainp, Y_testp = X_train+train_shift.values.reshape(train_shift.size,1), X_test+test_shift.values.reshape(test_shift.size,1), Y_train+train_shift.values.reshape(train_shift.size,1), Y_test + test_shift.values.reshape(test_shift.size,1)
# X_train = np.column_stack((X_train,Y_train))


def exec_time(start, msg):
	end = time.time()
	delta = end - start
	if(delta > 60): print("Tempo: " + str(delta/60.0) + " min [" + msg + "]")
	else: print("Tempo: " + str(int(delta)) + " s [" + msg + "]")

def generator_model(opt):
	model = Sequential()
	model.add(Dense(256, input_dim=120, kernel_initializer=RandomNormal(stddev=0.1)))
	model.add(LeakyReLU(0.4))
	model.add(Dense(512))
	model.add(LeakyReLU(0.4))
	model.add(Dense(1024))
	model.add(LeakyReLU(0.4))
	#model.add(Dense(121)) #output 120+previsao
	model.add(Dense(4)) #output so previsao
	model.add(Activation('tanh'))

	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def discriminator_model(opt):
	model = Sequential()

	model.add(Dense(1024, input_dim=124, kernel_initializer=RandomNormal(stddev=0.1)))
	model.add(LeakyReLU(0.4))
	model.add(Dropout(0.5))
	model.add(Dense(512))
	model.add(LeakyReLU(0.4))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(LeakyReLU(0.4))
	model.add(Dropout(0.5))
	model.add(Dense(4, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def gan_model(D, G, opt):
	D.trainable = False

	gan_input = Input(shape=(120,))
	x = concatenate([gan_input, G(gan_input)])
	gan_output = D(x)
	gan = Model(gan_input, gan_output)
	gan.compile(loss='binary_crossentropy', optimizer=opt)

	return gan

def train(X_train, generator, discriminator, GAN, epochs=6000, verbose_step=250, batch_size=128, output_dir='output'):
	print("*** Training", epochs, "epochs with batch size =", batch_size, "***")
	times = []
	d_lossses = []
	g_losses = []

	start_train = time.time()

	for e in range(epochs+1):
		start = time.time()

		noise = np.random.normal(0, 1, size=[batch_size, 120])
		input_data = X_train[np.random.randint(0, X_train.shape[0], size=2*batch_size)]
		imageBatch = input_data[:int(input_data.shape[0]/2)]
		#generator_data = np.delete(input_data[int(input_data.shape[0]/2):], -1, axis=1) # remove ultima coluna
		generator_data = np.delete(input_data[int(input_data.shape[0]/2):], [120,121,122,123], axis=1) #remove ultimas 4 colunas

		G_images = generator.predict(generator_data)
		G_images = np.column_stack((generator_data,G_images))
		X = np.concatenate([imageBatch, G_images])

		#y = np.zeros(2*batch_size)
		y = np.zeros((2*batch_size,4)) #OHLC target
		y[:batch_size] = 0.9


		discriminator.trainable = True
		d_loss = discriminator.train_on_batch(X, y)

		noise = np.random.normal(0, 1, size=[batch_size, 120])
		#y = np.ones(batch_size)
		y = np.ones((batch_size,4)) # OHLC target
		discriminator.trainable = False
		g_loss = GAN.train_on_batch(generator_data, y)

		d_lossses.append(d_loss)
		g_losses.append(g_loss)
		times.append(time.time() - start)
		
		plotGeneratedCandles("original_test", Y_testp[0:50,0], Y_testp[0:50,1], Y_testp[0:50,2], Y_testp[0:50,3], output_dir)
		if(e % verbose_step == 0):
			print(str(e) + ": d_loss =", d_loss, "| g_loss =", g_loss)
			#trainPredict = generator.predict( np.delete(X_train, -1, axis=1))
			trainPredict = generator.predict( np.delete(X_train, [120,121,122,123], axis=1))
			testPredict = generator.predict(X_test)
			new_predicted = testPredict+test_shift.values.reshape(test_shift.size,1)
			new_train_predicted= trainPredict+train_shift.values.reshape(train_shift.size,1)
			testScore = math.sqrt(mean_squared_error(new_predicted, Y_testp))
			trainScore = math.sqrt(mean_squared_error(new_train_predicted, Y_trainp))
			print("RMSE treino: %f"% trainScore)
			print("RMSE test: %f" % testScore)
			plotGeneratedCandles(e, new_predicted[0:50,0], new_predicted[0:50,1], new_predicted[0:50,2], new_predicted[0:50,3], output_dir)
			#plotGeneratedImages(e, generator, output_dir)
	

	
	exec_time(start_train, "Training")
	generate_graphics(times, d_lossses, g_losses, output_dir)

def generate_graphics(times, d_lossses, g_losses, output_dir):
	plt.close('all')
	x = np.linspace(0, len(times), len(times))

	plt.clf()
	plt.title("GAN MNIST - Exec time per epoch")
	plt.ylabel('seconds')
	plt.xlabel('epoch')
	plt.plot(x[1:], times[1:])
	plt.savefig(os.path.join(output_dir, 'times.png'))
	# plt.show()

	plt.clf()
	plt.title("GAN MNIST - D and G losses per epoch")
	plt.ylabel('loss(binary crossentropy)')
	plt.xlabel('epoch')
	plt.plot(x, d_lossses, 'b-', label="D loss")
	plt.plot(x, g_losses, 'g-', label="G loss")
	plt.savefig(os.path.join(output_dir, 'losses.png'))
	# plt.show()

def plotGeneratedImages(e, generator, output_dir, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, 100])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.close('all')

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(e) + '.png'))

def plotGeneratedCandles(e, opens, highs, lows, closes, output_dir):
	plt.close('all')
	fig, ax = plt.subplots()
	matplotlib.finance.candlestick2_ohlc(ax, opens, highs, lows, closes)
	#plt.show()
	plt.savefig(os.path.join(output_dir, str(e) + '.png'))
	

def main():
	if(len(sys.argv) > 1):
		folder = 'output_'+sys.argv[1]
		if os.path.exists(folder):
			shutil.rmtree(folder)
		os.makedirs(folder)
	
	
	opt = Adam(lr=0.0002, beta_1=0.5)

	generator = generator_model(opt)
	discriminator = discriminator_model(opt)
	GAN = gan_model(discriminator, generator, opt)

	train(X_train, generator, discriminator, GAN, output_dir=folder)


if __name__ == '__main__':
	start = time.time()
	main()
	exec_time(start, "All")