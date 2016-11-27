import cv2
import copy
import random
import tensorflow as tf
import numpy as np
from inputdata import Dataset
from neuralnet import MLP_net

def main():
	# define constants
	paths = ["tooldata/","nontooldata/"]	
	number_of_samples = 38000
	number_of_classes = 2
	image_width = 30
	image_height = 40
	image_type = "tiff"

	# prepare input data
	SurgicalData = Dataset(paths,number_of_samples,number_of_classes, image_width, image_height, image_type)
	
	# mlp method	
	epochs = 15
	batchsize = 100
	n_nodes = [image_width*image_height, 256, 256, 64, number_of_classes]
	MLP = MLP_net(number_of_samples,epochs,batchsize,n_nodes)
	MLP.train_neural_network(SurgicalData)
	MLP.test_neural_network(SurgicalData)
	
if __name__ == "__main__":
    main()
