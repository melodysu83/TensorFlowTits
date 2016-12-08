import cv2
import copy
import random
import tensorflow as tf
import numpy as np

# Class/Dataset
# desc/ This is a class for training and testing dataset
#           There are functions that extract sequential frames from videos
#           Then turn it into overlapped image patches of size 30x40
#           Because our classifier takes image inputs of 30x40 pixels
class Dataset:
        
                # This is an initialization function for object of this class.
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

                # This is a functions that creates one-hot label to the trainig set
                # in our case, since its binary classification, we will have
                # label = (1,0) for surgical tool image patches
                # label = (0,1) for non-surgical tool image patches
	def create_one_hot_label(self,class_label):
		label = []
		for num_class in range(self.number_of_classes):
			if num_class == class_label:
				label = label + [1]
			else:
				label = label + [0]
		return label

                # This is a functions that creates numeric label to the trainig set
                # in our case, since its binary classification, we will have
                # label = 0 for surgical tool image patches
                # label = 1 for non-surgical tool image patches
	def create_numeric_label(self,class_label):
		label = class_label
		return label

                # load image patches as training set
	def load_image_from_disk(self,class_label,filename):
		image_data = []
		path_of_image = self.paths[class_label] + str(filename) + self.image_type
		image = cv2.imread(path_of_image)

		for row in range(self.image_height):
			for col in range(self.image_width):
				intensity_value = image[row,col,0]
				image_data = image_data + [intensity_value]
		return image_data

                # generate training set and resize it into square patches,
                # so that it's easier to rotate without needing to change its shape
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

                # To prevend having to load the entire training set to RAM
                # we import only a batch of image patches each time
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

                # the function that gets / generates a list of image patches as test set
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

                # resize training set to square batches,
                # so its easier for rotation without needing to change its shape
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

               # resize testing set to square batches,
                # so its easier for rotation without needing to change its shape
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
