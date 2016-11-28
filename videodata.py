import numpy as np
import cv2
from matplotlib import pyplot as plt

class VideoData:

	def __init__(self, path, patch_size, overlap_size):
		self.path = 'surgicalvideo.avi'
		self.cap = cv2.VideoCapture(self.path)
		self.patch_size = patch_size
		self.overlap_size = overlap_size
		self.patch_locations_row = []
		self.patch_locations_col = []
		self.count = 0

	def resize(self,frame,scale):
		if scale > 0:
			height, width = frame.shape[:2]
			frame = frame[0:height-1,0:width/2-1]
			return cv2.resize(frame, (int(width/2*scale), int(height*scale)))
		else:
			print "Invalid resize request!"
			return frame
		
	def intensity(self, image):
		image_intensity = []
		rows, cols = image.shape[:2]
		for r in range(rows):
			for c in range(cols):
				value = image[r,c]
				image_intensity = image_intensity + [value]
		return image_intensity


	def patchify(self, image, square = False, reshape_size = 28):
		count = 0
		first_loop = True
		height, width = image.shape[:2]
	
		if self.patch_size[0] > height or self.patch_size[1] > width:
			print "Image too small to patchify."
			return count,None


		self.patch_locations_row = []
		self.patch_locations_col = []

		row = self.patch_size[0]-1 # botthom right pixel
		col = self.patch_size[1]-1

		while True:	
			image_patch = image[row- self.patch_size[0]+1:row+1, col- self.patch_size[1]+1:col+1]
			
			if square:
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

			elif col < width-1:   		  # shift right to edge       
				col = width - 1

			elif row+self.patch_size[1] < height:  # shift down
				col = self.patch_size[1] - 1
				row = row + self.patch_size[1]

			elif row < height-1:
				col = self.patch_size[1] - 1
				row = height - 1
			else:
				break;		

		self.count = count
		return count,image_patches


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



	


