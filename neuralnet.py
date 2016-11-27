import cv2
import copy
import random
import tensorflow as tf
import numpy as np
from inputdata import Dataset

# multi-layer perceptron neural network
class MLP_net:
	def __init__(self,number_of_samples,epochs,batchsize,n_nodes):
		self.epochs = epochs
		self.batchsize = batchsize
		self.n_nodes = n_nodes
		self.number_of_samples = number_of_samples
		self.number_of_classes = self.n_nodes[4]
		# input, hiddenL1, hiddenL2, output

		self.x = tf.placeholder('float',[ None,n_nodes[0]]) # our input data (28*28=784 pixels per data)
		self.y = tf.placeholder('float') # the label of our data
		self.weights = {
			'h1': tf.Variable(tf.random_normal([self.n_nodes[0], self.n_nodes[1]],name = 'W_h1')),
			'h2': tf.Variable(tf.random_normal([self.n_nodes[1], self.n_nodes[2]],name = 'W_h2')),
			'h3': tf.Variable(tf.random_normal([self.n_nodes[2], self.n_nodes[3]],name = 'W_h3')),
			'output': tf.Variable(tf.random_normal([self.n_nodes[3], self.n_nodes[4]],name = 'W_output'))
		}
		self.biases = {
			'h1': tf.Variable(tf.random_normal([self.n_nodes[1]],name = 'b_h1')),
			'h2': tf.Variable(tf.random_normal([self.n_nodes[2]],name = 'b_h2')),
			'h3': tf.Variable(tf.random_normal([self.n_nodes[3]],name = 'b_h3')),
			'output': tf.Variable(tf.random_normal([self.n_nodes[4]],name = 'b_output'))
		}

	def neural_network_model(self,data):
		# (input_data * weights) + biases
		n_layer1 = tf.add(tf.matmul(data,self.weights['h1']),self.biases['h1'])
		n_layer1 = tf.nn.relu(n_layer1)
		n_layer2 = tf.add(tf.matmul(n_layer1,self.weights['h2']),self.biases['h2'])
		n_layer2 = tf.nn.relu(n_layer2)
		n_layer3 = tf.add(tf.matmul(n_layer2,self.weights['h3']),self.biases['h3'])
		n_layer3 = tf.nn.relu(n_layer3)
		output = tf.add(tf.matmul(n_layer3,self.weights['output']),self.biases['output'])
		return output
		

	def train_neural_network(self, my_data):
		print "training phase"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare saver object
		saver = tf.train.Saver(tf.all_variables())
				
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())

			# training stage
			for epoch in range(self.epochs):
				epoch_loss = 0
				for batch_number in range(int(self.number_of_samples/self.batchsize)):
					batch_x, batch_y = my_data.images_to_data_batch(batch_number,self.batchsize)
					_, c = sess.run([optimizer, cost], feed_dict = {self.x: batch_x, self.y: batch_y})
					epoch_loss += c
				print('Epoch', epoch, 'completed out of', self.epochs,' loss:',epoch_loss)

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			
			# validation stage
			test_data,test_label = my_data.get_test_set()
			print('Accuracy:',accuracy.eval({self.x:test_data, self.y:test_label}))

			# save model value
			saver.save(sess, 'model/MLP_model.chk')

	def test_neural_network(self, my_data):
		print "testing phase"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare saver object
		saver = tf.train.Saver([self.weights['h1'],self.weights['h2'],self.weights['h3'],self.weights['output'],self.biases['h1'],self.biases['h2'],self.biases['h3'],self.biases['output']])
				
		with tf.Session() as sess:
			saver.restore(sess,'model/MLP_model.chk')

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			
			# validation stage
			test_data,test_label = my_data.get_test_set()
			print('Accuracy:',accuracy.eval({self.x:test_data, self.y:test_label}))
