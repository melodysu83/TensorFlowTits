import cv2
import copy
import random
import tensorflow as tf
import numpy as np
from inputdata import Dataset
from neuralnet import MLP_net
from neuralnet import CNN_net

class Menu:

	def Initial(self):
		# Task related modification can be changed here:
		self.paths = ["tooldata/","nontooldata/"]	
		self.number_of_samples = 38000
		self.number_of_classes = 2
		self.image_width = 30
		self.image_height = 40
		self.image_type = "tiff"
		

	def __init__(self, training=True, testing=True):		
		self.training = training
		self.testing = testing

		self.MLP = 0   # multi-layer-perceptron
		self.RNN = 1   # recurrent-neural-network
		self.CNN = 2   # convolution-neural-network
		
		self.paths = None	
		self.number_of_samples = None
		self.number_of_classes =  None
		self.image_width =  None
		self.image_height =  None
		self.image_type =  None
		self.Initial()
		self.Data = Dataset(self.paths,self.number_of_samples,self.number_of_classes, self.image_width, self.image_height, self.image_type)


	def MLP_Process(self, epochs,batchsize): # multi-layer-perceptron
		# mlp method
		# (with model accuracy 0.90263158)
		print "mlp method"
		
		n_nodes = [self.image_width*self.image_height, 256, 256, 64, self.number_of_classes]

		MLP = MLP_net(self.number_of_samples,epochs,batchsize,self.n_nodes)

		if self.training:
			MLP.train_neural_network(self.Data)

		elif self.testing:
			MLP.test_neural_network(self.Data)


	def RNN_Process(epochs,batchsize,keep_rate): # recurrent-neural-network
		# rnn method
		# (with model accuracy ?)
		print "rnn method"


	def CNN_Process(self, epochs,batchsize,keep_rate): # convolution-neural-network
		# cnn method
		# (with model accuracy ?)
		print "cnn method"

		n_nodes = [1,32,64,1024,self.number_of_classes] # [n_features[0], n_features[1], n_features[2], fc_nodes, n_classes]
		sizes = [5, 7, 28] # [window_size, fc_imgsize, reshape_size]

		CNN = CNN_net(self.number_of_samples,epochs,batchsize,n_nodes,sizes,keep_rate)
	
	
		if self.training:
			CNN.train_neural_network(self.Data)

		elif self.testing:
			CNN.test_neural_network(self.Data)
