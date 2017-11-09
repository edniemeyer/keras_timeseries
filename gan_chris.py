# original reference: https://github.com/Zackory/Keras-MNIST-GAN

import sys
import os
import shutil
from keras.datasets import mnist
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Sequential
from keras.layers import Reshape, Dense, Dropout, Flatten, Conv2D, LeakyReLU, Activation, Input
import matplotlib.pyplot as plt
import time

np.random.seed(42)

def exec_time(start, msg):
	end = time.time()
	delta = end - start
	if(delta > 60): print("Tempo: " + str(delta/60.0) + " min [" + msg + "]")
	else: print("Tempo: " + str(int(delta)) + " s [" + msg + "]")

def generator_model(opt):
	model = Sequential()
	model.add(Dense(256, input_dim=100, kernel_initializer=RandomNormal(stddev=0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1024))
	model.add(LeakyReLU(0.2))
	model.add(Dense(784))
	model.add(Activation('tanh'))

	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def discriminator_model(opt):
	model = Sequential()

	model.add(Dense(1024, input_dim=784, kernel_initializer=RandomNormal(stddev=0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def gan_model(D, G, opt):
	D.trainable = False

	gan_input = Input(shape=(100,))
	gan_output = D(G(gan_input))
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

		noise = np.random.normal(0, 1, size=[batch_size, 100])
		imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

		G_images = generator.predict(noise)
		X = np.concatenate([imageBatch, G_images])

		y = np.zeros(2*batch_size)
		y[:batch_size] = 0.9

		discriminator.trainable = True
		d_loss = discriminator.train_on_batch(X, y)

		noise = np.random.normal(0, 1, size=[batch_size, 100])
		y = np.ones(batch_size)
		discriminator.trainable = False
		g_loss = GAN.train_on_batch(noise, y)

		d_lossses.append(d_loss)
		g_losses.append(g_loss)
		times.append(time.time() - start)

		if(e % verbose_step == 0):
			print(str(e) + ": d_loss =", d_loss, "| g_loss =", g_loss)
			plotGeneratedImages(e, generator, output_dir)

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

def main():
	if(len(sys.argv) > 1):
		folder = 'output_'+sys.argv[1]
		if os.path.exists(folder):
			shutil.rmtree(folder)
		os.makedirs(folder)

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = (X_train.astype(np.float32)-127.5)/127.5
	X_train = X_train.reshape(60000, 784)

	opt = Adam(lr=0.0002, beta_1=0.5)

	generator = generator_model(opt)
	discriminator = discriminator_model(opt)
	GAN = gan_model(discriminator, generator, opt)

	train(X_train, generator, discriminator, GAN, output_dir=folder)

if __name__ == '__main__':
	start = time.time()
	main()
	exec_time(start, "All")