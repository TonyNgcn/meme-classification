import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from PIL import Image
import random

class_num = 10
image_size = 32
img_channels = 3


def rgb_normalization(filename):
	gifimg = Image.open(filename)
	if (gifimg.mode != "RGB"):
		gifimg = gifimg.convert('RGB')
	out=gifimg.resize((32, 32), Image.ANTIALIAS)
	name = filename[:-4] + '_pred.jpg'
	out.save(name)
	return name


def get_rgb_vector(imgPath):
	img = cv.imread(imgPath)
	# img=cv.resize(img,(32,32))
	B, G, R = cv.split(img)
	B = np.reshape(B, (1, 1024))
	G = np.reshape(G, (1, 1024))
	R = np.reshape(R, (1, 1024))
	vector = np.concatenate((R, G, B), axis=1)
	return vector


def prepare_data(imgPath):
	imgPath=rgb_normalization(imgPath)
	data = []
	img=cv.imread(imgPath)
	data.append(img)
	data = np.array(data)
	return data


def _random_crop(batch, crop_shape, padding=None):
	oshape = np.shape(batch[0])

	if padding:
		oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
	new_batch = []
	npad = ((padding, padding), (padding, padding), (0, 0))
	for i in range(len(batch)):
		new_batch.append(batch[i])
		if padding:
			new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
			                          mode='constant', constant_values=0)
		nh = random.randint(0, oshape[0] - crop_shape[0])
		nw = random.randint(0, oshape[1] - crop_shape[1])
		new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
		               nw:nw + crop_shape[1]]
	return new_batch


def _random_flip_leftright(batch):
	for i in range(len(batch)):
		if bool(random.getrandbits(1)):
			batch[i] = np.fliplr(batch[i])
	return batch


def color_preprocessing(x):
	x = x.astype('float32')
	x[:, :, :, 0] = (x[:, :, :, 0] - np.mean(x[:, :, :, 0])) / np.std(x[:, :, :, 0])
	x[:, :, :, 1] = (x[:, :, :, 1] - np.mean(x[:, :, :, 1])) / np.std(x[:, :, :, 1])
	x[:, :, :, 2] = (x[:, :, :, 2] - np.mean(x[:, :, :, 2])) / np.std(x[:, :, :, 2])

	return x


def data_augmentation(batch):
	batch = _random_flip_leftright(batch)
	batch = _random_crop(batch, [32, 32], 4)
	return batch
