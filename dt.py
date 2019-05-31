import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import random
class_num = 10
image_size = 32
img_channels = 3


def get_rgb_vector(imgPath):
	img = cv.imread(imgPath)
	# img=cv.resize(img,(32,32))
	B, G, R = cv.split(img)
	B = np.reshape(B, (1, 1024))
	G = np.reshape(G, (1, 1024))
	R = np.reshape(R, (1, 1024))
	vector = np.concatenate((R, G, B), axis=1)
	return vector


def prepare_data():
	dir='./ok'
	labels=[]
	data=[]
	for i,imgpath in enumerate(os.listdir(dir)):
		# if i==0:
		# 	data=get_rgb_vector(dir+'/'+imgpath)
		# else:
		# 	data = np.concatenate((data,get_rgb_vector(dir+'/'+imgpath)), axis=0)

		data.append(cv.imread(dir+'/'+imgpath))
		labels.append(int(imgpath[:2]))
	labels=np.array(labels)
	data=np.array(data)

	train_data, test_data, train_labels, test_labels = train_test_split(data,labels,test_size=0.2)

	onehot=OneHotEncoder()
	train_labels=onehot.fit_transform(train_labels.reshape(-1,1)).toarray()
	test_labels=onehot.fit_transform(test_labels.reshape(-1,1)).toarray()

	return train_data, train_labels, test_data, test_labels


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


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch