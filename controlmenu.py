import cv2
import copy
import random
import tensorflow as tf
import numpy as np
from inputdata import Dataset
from videodata import VideoData
from neuralnet import MLP_net
from neuralnet import CNN_net
from neuralnet import RNN_net

# Class/ Menu
# desc/ This is a class for tcontemplating for proper actions according to user selections
#           There are processes that corresponds to 3 types of neural network models.
#           MLP_Porcess, RNN_Process and CNN_Process
#           each process contains three possible actions depending on the user selection,
#           you can be in either the testing, training, or real world implementation phase
class Menu:
                # This is another initialization function relating to data preparation.
	def Initial(self):
		# Task related modification can be changed here:
		self.paths = ["tooldata/","nontooldata/"]	
		self.number_of_samples = 38000
		self.number_of_classes = 2
		self.image_width = 30
		self.image_height = 40
		self.image_type = "tiff"

		self.video_path = "surgicalvideo.avi"
		self.video_scale = 0.5
		self.video_overlap_size = [5,5]
		self.video_patch_size = [self.image_height,self.image_width]


                # This is an initialization function for object of this class.
	def __init__(self, mode = 0):	
	
		self.mode = mode # (0: training, 1: testing, 2: real world implementation)  

		self.MLP = 0   # multi-layer-perceptron
		self.RNN = 1   # recurrent-neural-network
		self.CNN = 2   # convolution-neural-network
		
		self.training = 0     # training phase
		self.testing = 1      # testing phase
		self.real_world = 2   # real world implementation

		self.paths = None	
		self.number_of_samples = None
		self.number_of_classes =  None
		self.image_width =  None
		self.image_height =  None
		self.image_type =  None
		self.video_path = None
		self.video_scale = None
		self.video_overlap_size = None
		self.video_patch_size = None

		self.Initial()
		self.Data = Dataset(self.paths,self.number_of_samples,self.number_of_classes, self.image_width, self.image_height, self.image_type)
		self.VidData = VideoData(self.video_path,self.video_patch_size,self.video_overlap_size)

	def MLP_Process(self, epochs, batchsize): # multi-layer-perceptron
		# mlp method
		# (with model accuracy 0.90263158)
		print "mlp method"
		
		n_nodes = [self.image_width*self.image_height, 256, 256, 64, self.number_of_classes]

		MLP = MLP_net(self.number_of_samples,epochs,batchsize,n_nodes)

		if self.mode == self.training:
			MLP.train_neural_network(self.Data)

		elif self.mode == self.testing:
			MLP.test_neural_network(self.Data)

		elif self.mode == self.real_world:
			MLP.real_world_neural_network(self.VidData, self.video_scale)


	def RNN_Process(self, epochs, batchsize, chunk): # recurrent-neural-network
		# rnn method
		# (with model accuracy 0.92236841)
		print "rnn method"
		n_nodes = [chunk*chunk, self.number_of_classes]
		sizes = [chunk, chunk, 128] # [chunk_size, n_chunks, rnn_size]
		
		RNN = RNN_net(self.number_of_samples,epochs,batchsize,n_nodes,sizes)
		
		if self.mode == self.training:
			RNN.train_neural_network(self.Data)

		elif self.mode == self.testing:
			RNN.test_neural_network(self.Data)	
	
		elif self.mode == self.real_world:
			RNN.real_world_neural_network(self.VidData, self.video_scale)



	def CNN_Process(self, epochs, batchsize, keep_rate): # convolution-neural-network
		# cnn method
		# (with model accuracy 0.98815787)
		print "cnn method"

		n_nodes = [1,32,64,1024,self.number_of_classes] # [n_features[0], n_features[1], n_features[2], fc_nodes, n_classes]
		sizes = [5, 7, 28] # [window_size, fc_imgsize, reshape_size]

		CNN = CNN_net(self.number_of_samples,epochs,batchsize,n_nodes,sizes,keep_rate)
	
	
		if self.mode == self.training:
			CNN.train_neural_network(self.Data)

		elif self.mode == self.testing:
			CNN.test_neural_network(self.Data)

		elif self.mode == self.real_world:
			CNN.real_world_neural_network(self.VidData, self.video_scale)

		
