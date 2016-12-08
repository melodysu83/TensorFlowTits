import numpy as np
import cv2
from matplotlib import pyplot as plt

# Class/ VideoData
# desc/ This is a class for video data extraction
#           There are functions that extract sequential frames from videos
#           Then turn it into overlapped image patches of size 30x40
#           Because our classifier takes image inputs of 30x40 pixels
class VideoData:

                # This is an initialization function for object of this class.
	def __init__(self, path, patch_size, overlap_size):
		self.path = 'surgicalvideo.avi'
		self.cap = cv2.VideoCapture(self.path)
		self.patch_size = patch_size
		self.overlap_size = overlap_size
		self.patch_locations_row = []
		self.patch_locations_col = []
		self.count = 0

                # We resize the video frame by 1/2 on both vertical and horizontal direction
                # for the sake of faster computation
	def resize(self,frame,scale):
		if scale > 0:
			height, width = frame.shape[:2]
			
			# Take only the left half of the stereo image
			frame = frame[0:height-1,0:width/2-1]
			
			# Width/2 because it's a stereo image pair,
			# and we're taking only the left half.
			return cv2.resize(frame, (int(width/2*scale), int(height*scale)))
		else:
			print "Invalid resize request!"
			return frame

	# This is a function that stores image intensity values
	# of an image patch into a long array of values
	def intensity(self, image):
		image_intensity = []
		rows, cols = image.shape[:2]
		
		for r in range(rows):
			for c in range(cols):
				value = image[r,c]
				image_intensity = image_intensity + [value]
		return image_intensity


                # This function that splits a big image frame into small overlapped image patches 
	def patchify(self, image, square = False, reshape_size = 28):
		count = 0
		first_loop = True
		height, width = image.shape[:2] # get image height and width

                                # can't patchify if the whole image is smaller than a patch
		if self.patch_size[0] > height or self.patch_size[1] > width:
			print "Image too small to patchify."
			return count,None

		self.patch_locations_row = []
		self.patch_locations_col = []

		row = self.patch_size[0]-1 # bottom right pixel
		col = self.patch_size[1]-1

		while True:	
			image_patch = image[row- self.patch_size[0]+1:row+1, col- self.patch_size[1]+1:col+1]
			
			if square:  # require square shaped image patches
				image_patch = cv2.resize(image_patch,(reshape_size,reshape_size))

			value =  np.array([self.intensity(image_patch)])
			
			if first_loop:
				first_loop = False
				image_patches = value
			else:
				image_patches = np.concatenate((image_patches,value))

			self.patch_locations_row = self.patch_locations_row+[row]
			self.patch_locations_col = self.patch_locations_col+[col]
			count = count + 1

			if col+self.patch_size[1] < width:     # shift right
				col = col + self.patch_size[1]

			elif col < width-1:   # shift right to edge       
				col = width - 1

			elif row+self.patch_size[1] < height:  # shift down
				col = self.patch_size[1] - 1
				row = row + self.patch_size[1]

			elif row < height-1:  # shift down to edge
				col = self.patch_size[1] - 1
				row = height - 1
			else:
				break;		

		self.count = count
		return count,image_patches

                # This function prints out the detection result of surgical instruments
                # and mark them as rectangular boxes
	def draw_result(self, frame, fire_index):
		# do something (draw rect) ...
		for index in range(self.count):
			if fire_index[index] == 0: # is the surgical tool!
				y2 = self.patch_locations_row[index]
				x2 = self.patch_locations_col[index]
				y1 = y2 - self.patch_size[0] + 1
				x1 = x2 - self.patch_size[1] + 1
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return frame



	



