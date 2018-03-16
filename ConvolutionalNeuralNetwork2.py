from Othello import *

import numpy as np
import os
import random
import sys
import tensorflow as tf
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Logging/Warnings


class ConvolutionalNeuralNetwork2:
	def __init__(self, input_depth=None, num_layers=2, num_filters=None, dropout_rate=0.5, verbose=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.board_size = OthelloConstants.BOARD_SIZE
			self.input_depth = input_depth
			self.is_training = tf.placeholder(tf.bool)
			
			self.x = tf.placeholder(tf.float32, [None, self.board_size, self.board_size, self.input_depth], name="x")
			self.y = tf.placeholder(tf.float32, [None, 1], name="y")
			self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

			layers = [self.x]
			for i in range(num_layers):
				layers.append(tf.layers.dropout(
					inputs = tf.layers.conv2d(
						inputs = layers[-1],
						filters = num_filters[i],
						kernel_size = 3,
						padding = "same",
						kernel_initializer = tf.contrib.layers.xavier_initializer(),
						bias_initializer = tf.zeros_initializer(),#tf.contrib.layers.xavier_initializer(),
						activation = tf.nn.elu,
						name = "conv{0}".format(i+1)
					),
					rate = dropout_rate,
					training = self.is_training,
					name = "dropout_conv{0}".format(i+1)
				))

			layers.append(tf.layers.conv2d(
				inputs = layers[-1],
				filters = 1,
				kernel_size = 1,
				padding = "same",
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),#tf.contrib.layers.xavier_initializer(),
				activation = tf.nn.elu,
				name = "final_conv"
			))

			dense1 = tf.layers.dropout(
				inputs = tf.layers.dense(
					inputs = tf.reshape(layers[-1], [-1, self.tuple_product(layers[-1].shape)]),
					units = 1024,
					kernel_initializer = tf.contrib.layers.xavier_initializer(),
					bias_initializer = tf.zeros_initializer(),
					activation = tf.nn.elu,
					name = "dense1"
				),
				rate = dropout_rate,
				training = self.is_training,
				name = "dropout_dense{0}".format(i+1)
			)

			dense2 = tf.layers.dense(
				inputs = dense1,
				units = 1024,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.tanh,
				name = "dense2"
			)

			dense3 = tf.layers.dense(
				inputs = dense2,
				units = 1,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.tanh,
				name = "dense3"
			)

			self.output = tf.reshape(dense3, [-1, 1])
			self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.output))
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, tf.sign(self.output)), tf.float32))
			self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			self.session = tf.Session()
			self.session.run(tf.global_variables_initializer())

	def tuple_product(self, tup):
		product = 1
		for elem in tup:
			try:
				product *= int(elem)
			except:
				pass
		return product

	def save(self, filename):
		with self.graph.as_default():
			self.saver = tf.train.Saver()
			self.saver.save(self.session, filename)

	def load(self, filename):
		with self.graph.as_default():
			self.saver = tf.train.Saver()
			self.saver.restore(self.session, filename)

	def predict(self, data, softmax=True):
		data_reshaped = np.reshape(data, [-1, self.board_size, self.board_size, self.input_depth])
		return self.session.run(self.output, feed_dict = {self.x: data_reshaped, self.is_training: False})

	def generate_batches(self, x, y, batch_size):
		data = list(zip(x, y))
		random.shuffle(data)
		num_batches = int(np.ceil(len(x)/batch_size))
		for i in range(num_batches):
			yield tuple(zip(*data[batch_size*i:batch_size*(i+1)]))

	def evaluate_performance(self, x_data, y_data):
		return self.session.run((self.loss, self.accuracy), feed_dict = {
			self.x: x_data,
			self.y: y_data,
			self.is_training: False
		})

	def train(self, x_train, y_train,
					x_valid, y_valid,
				    batch_size = 20,
				    num_epochs = 10,
				    learning_rate = 1e-4,
				    model_name = None):
		
		print ("Training Set Size: {0}, Validation Set Size: {1}".format(len(y_train), len(y_valid)))
		lowest_loss = float("inf")

		for epoch in range(num_epochs):
			print('Epoch #{0} out of {1}'.format(epoch+1, num_epochs))
			progbar = Progbar(target=int(np.ceil(len(x_train)/batch_size)))

			for batch, (x_batch, y_batch) in enumerate(self.generate_batches(x_train, y_train, batch_size)):
				train_loss, train_accuracy = self.evaluate_performance(x_batch, y_batch)
				progbar.update(batch+1, [("Loss", train_loss), ("Accuracy", train_accuracy)])

				self.session.run(self.train_step, feed_dict = {
					self.x: x_batch,
					self.y: y_batch,
					self.learning_rate: learning_rate,
					self.is_training: True
				})

			if len(x_valid) > 0:
				valid_loss, valid_accuracy = self.evaluate_performance(x_valid, y_valid)
				marker = ""
				if model_name is not None and valid_loss <= lowest_loss:
					lowest_loss = valid_loss
					self.save(model_name)
					marker = "*"

				print('Validation Set - Loss: {0:.4f}, Accuracy: {1:.4f} {2}\n'.format(valid_loss, valid_accuracy, marker))



class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)
