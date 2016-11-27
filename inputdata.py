import cv2
import copy
import random
import tensorflow as tf
import numpy as np
class Dataset:
	def __init__(self,paths,number_of_samples,number_of_classes,image_width, image_height, image_type):
		self.paths = paths
		self.number_of_samples = number_of_samples
		self.number_of_classes = number_of_classes
		self.image_width = image_width
		self.image_height = image_height
		self.image_type = "." + image_type
		self.test_set = []
		self.test_set_size = number_of_samples/100
		for i in range(self.test_set_size):
			self.test_set = self.test_set + [random.randint(1,number_of_samples)]


	def create_one_hot_label(self,class_label):
		label = []
		for num_class in range(self.number_of_classes):
			if num_class == class_label:
				label = label + [1]
			else:
				label = label + [0]
		return label


	def create_numeric_label(self,class_label):
		label = class_label
		return label


	def load_image_from_disk(self,class_label,filename):
		image_data = []
		path_of_image = self.paths[class_label] + str(filename) + self.image_type
		image = cv2.imread(path_of_image)

		for row in range(self.image_height):
			for col in range(self.image_width):
				intensity_value = image[row,col,0]
				image_data = image_data + [intensity_value]
		return image_data

	def load_square_image_from_disk(self,class_label,filename, reshape_size):
		image_data = []
		path_of_image = self.paths[class_label] + str(filename) + self.image_type
		image = cv2.imread(path_of_image)
		image = cv2.resize(image, (reshape_size, reshape_size)) 

		for row in range(reshape_size):
			for col in range(reshape_size):
				intensity_value = image[row,col,0]
				image_data = image_data + [intensity_value]
		return image_data

	def images_to_data_batch(self,batch_number,batch_size):
		first_loop = True
		for _ in range(batch_size):
			filename = str(batch_number*batch_size + 1)
			for class_label in range(self.number_of_classes):
				image_data = np.array([self.load_image_from_disk(class_label,filename)])
				image_label = np.array([self.create_one_hot_label(class_label)])
				#image_label = np.array([self.create_numeric_label(class_label)])
				
				if first_loop:
					first_loop = False
					images_data = image_data
					images_label = image_label
				else:
					images_data = np.concatenate((images_data,image_data))
					images_label = np.concatenate((images_label,image_label))
		return images_data,images_label


	def get_test_set(self):
		first_loop = True
		for index in range(self.test_set_size):
			filename = self.test_set[index]
			for class_label in range(self.number_of_classes):
				image_data = np.array([self.load_image_from_disk(class_label,filename)])
				image_label = np.array([self.create_one_hot_label(class_label)])
				#image_label = np.array([self.create_numeric_label(class_label)])

				if first_loop:
					first_loop = False
					images_data = image_data
					images_label = image_label
				else:
					images_data = np.concatenate((images_data,image_data))
					images_label = np.concatenate((images_label,image_label))
		return images_data,images_label


	def images_to_square_data_batch(self,batch_number,batch_size,reshape_size):
		first_loop = True
		for _ in range(batch_size):
			filename = str(batch_number*batch_size + 1)
			for class_label in range(self.number_of_classes):
				image_data = np.array([self.load_square_image_from_disk(class_label,filename, reshape_size)])
				image_label = np.array([self.create_one_hot_label(class_label)])
				#image_label = np.array([self.create_numeric_label(class_label)])
				
				if first_loop:
					first_loop = False
					images_data = image_data
					images_label = image_label
				else:
					images_data = np.concatenate((images_data,image_data))
					images_label = np.concatenate((images_label,image_label))
		return images_data,images_label


	def get_square_test_set(self,reshape_size):
		first_loop = True
		for index in range(self.test_set_size):
			filename = self.test_set[index]
			for class_label in range(self.number_of_classes):
				image_data = np.array([self.load_square_image_from_disk(class_label,filename, reshape_size)])
				image_label = np.array([self.create_one_hot_label(class_label)])
				#image_label = np.array([self.create_numeric_label(class_label)])

				if first_loop:
					first_loop = False
					images_data = image_data
					images_label = image_label
				else:
					images_data = np.concatenate((images_data,image_data))
					images_label = np.concatenate((images_label,image_label))
		return images_data,images_label
