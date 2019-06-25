import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from dt_predict import *
from matplotlib import pyplot as plt
import keras.layers
import numpy as np
import cv2 as cv

predict_imgPath = 'test_img.jpg'
weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 8  # how many split ?
blocks = 3  # res_block ! (split + transition)
depth = 64  # out channel

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""
prediction_dict = ['小猪佩奇', '蘑菇头', '可爱柴犬', '猥琐猫', '猫', '笑', '熊本熊', '熊猫头', '文字', '长草颜']
reduction_ratio = 4
class_num = 10
batch_size = 1

# iteration = 391
# 128 * 391 ~ 50,000

iteration = 1
# 1*1~1

test_iteration = 10

total_epochs = 100


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
	with tf.name_scope(layer_name):
		network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
		                           padding=padding)
		return network


def Global_Average_Pooling(x):
	return global_avg_pool(x, name='Global_avg_pooling')


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
	return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
	with arg_scope([batch_norm],
	               scope=scope,
	               updates_collections=None,
	               decay=0.9,
	               center=True,
	               scale=True,
	               zero_debias_moving_mean=True):
		return tf.cond(training,
		               lambda: batch_norm(inputs=x, is_training=training, reuse=None),
		               lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
	return tf.nn.relu(x)


def Sigmoid(x):
	return tf.nn.sigmoid(x)


def Concatenation(layers):
	return tf.concat(layers, axis=3)


def Fully_connected(x, units=class_num, layer_name='fully_connected'):
	with tf.name_scope(layer_name):
		return tf.layers.dense(inputs=x, use_bias=False, units=units)


class SE_ResNeXt():
	def __init__(self, x, training):
		self.training = training
		self.model = self.Build_SEnet(x)

	def first_layer(self, x, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope + '_conv1')
			x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
			x = Relu(x)

			return x

	def transform_layer(self, x, stride, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=depth, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
			x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
			x = Relu(x)

			x = conv_layer(x, filter=depth, kernel=[3, 3], stride=stride, layer_name=scope + '_conv2')
			x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
			x = Relu(x)
			return x

	def transition_layer(self, x, out_dim, scope):
		with tf.name_scope(scope):
			x = conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
			x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
			# x = Relu(x)

			return x

	def split_layer(self, input_x, stride, layer_name):
		with tf.name_scope(layer_name):
			layers_split = list()
			for i in range(cardinality):
				splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
				layers_split.append(splits)

			return Concatenation(layers_split)

	def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
		with tf.name_scope(layer_name):
			squeeze = Global_Average_Pooling(input_x)

			excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
			excitation = Relu(excitation)
			excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
			excitation = Sigmoid(excitation)

			excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
			scale = input_x * excitation

			return scale

	def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
		# split + transform(bottleneck) + transition + merge
		# input_dim = input_x.get_shape().as_list()[-1]

		for i in range(res_block):
			input_dim = int(np.shape(input_x)[-1])

			if input_dim * 2 == out_dim:
				flag = True
				stride = 2
				channel = input_dim // 2
			else:
				flag = False
				stride = 1

			x = self.split_layer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
			x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))
			x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio,
			                                  layer_name='squeeze_layer_' + layer_num + '_' + str(i))

			if flag is True:
				pad_input_x = Average_pooling(input_x)
				pad_input_x = tf.pad(pad_input_x,
				                     [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
			else:
				pad_input_x = input_x

			input_x = Relu(x + pad_input_x)

		return input_x

	def Build_SEnet(self, input_x):
		# only cifar10 architecture

		input_x = self.first_layer(input_x, scope='first_layer')

		x = self.residual_layer(input_x, out_dim=64, layer_num='1')
		x = self.residual_layer(x, out_dim=128, layer_num='2')
		x = self.residual_layer(x, out_dim=256, layer_num='3')

		x = Global_Average_Pooling(x)
		x = flatten(x)

		x = Fully_connected(x, layer_name='final_fully_connected')
		return x


predict_x = prepare_data(predict_imgPath)
predict_x = color_preprocessing(predict_x)

# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_ResNeXt(x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())
predict_x = data_augmentation(predict_x)
x_feed_dict = {
	x: predict_x,
	training_flag: False
}
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	ckpt = tf.train.get_checkpoint_state('./model')
	if ckpt and ckpt.model_checkpoint_path:
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		saver.restore(sess, ckpt.model_checkpoint_path)
		print('Loading success')
	else:
		print('No checkpoint')
	prediction = sess.run(logits, feed_dict=x_feed_dict)
max_index = np.argmax(prediction)
result = prediction_dict[max_index]
print(result)
img = cv.imread(predict_imgPath)
fig = plt.figure()
plt.imshow(img)
fig.show()
