import cv2
import copy
import random
import tensorflow as tf
import numpy as np
from inputdata import Dataset
from tensorflow.python.ops import rnn, rnn_cell

# Class/MLP_net
# breif/ multi-layer perceptron neural network
# desc/ This is a class for defining our MLP network
#           which saves/creates the structure of the model
#           define the training procedure, testing procedure
#           and real world implementation procedure
class MLP_net:
                # This is the initialization of the class objects
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

                # This function defines our MLP model structure
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
		
                # This is a function that tests how MLP classifier works
                # in training phase for our training data set
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
		sess.close()

                # This is a function that tests how MLP classifier works
                # in testing phase for our testing data set
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
		sess.close()

                # This is a function that tests how MLP classifier works
                # for identifying surgical instruments in real time surgical operations video
	def real_world_neural_network(self, my_data, scale):
		print "real world implementation"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare neural network object
		saver = tf.train.Saver([self.weights['h1'],self.weights['h2'],self.weights['h3'],self.weights['output'],self.biases['h1'],self.biases['h2'],self.biases['h3'],self.biases['output']])
		
		with tf.Session() as sess:
			saver.restore(sess,'model/MLP_model.chk')

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			predict = tf.argmax(prediction,1)
			
			# validation stage
			while my_data.cap.isOpened():

				# prepare video data object
				ret, src = my_data.cap.read() # faster!
				ret, src = my_data.cap.read()
				frame = my_data.resize(src,scale)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# extract input data for neural network
				_,test_data = my_data.patchify(gray,False) # image,square,reshape_size		

				# obtain neural network output							
				fire_index = predict.eval(feed_dict={self.x: test_data}, session=sess)

				# draw it on the gray image
				gray = my_data.draw_result(gray, fire_index)
				
				# display classification result
				cv2.imshow('frame',frame)
			    	cv2.imshow('gray',gray)
			    	if cv2.waitKey(1) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
		sess.close()
		cap.release()
		cv2.destroyAllWindows()


# Class/RNN_net
# breif/ recurrent neural network
# desc/ This is a class for defining our RNN network
#           which saves/creates the structure of the model
#           define the training procedure, testing procedure
#           and real world implementation procedure
class RNN_net:
                # This is the initialization of the class objects
	def __init__(self,number_of_samples,epochs,batchsize,n_nodes,sizes):
		self.epochs = epochs
		self.batchsize = batchsize
		self.number_of_samples = number_of_samples
		self.number_of_classes = n_nodes[1]

		self.n_nodes = [n_nodes[0],n_nodes[1]]
		self.chunk_size = sizes[0]
		self.reshape_size = sizes[0] 		
		self.n_chunks = sizes[1]
		self.rnn_size = sizes[2]
		
		self.x = tf.placeholder('float',[ None, self.n_chunks, self.chunk_size]) # our input data (28*28=784 pixels per data)
		self.y = tf.placeholder('float') # the label of our data

		self.layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_nodes[1]]), name='weights'),
			      'biases': tf.Variable(tf.random_normal([self.n_nodes[1]]), name='biases')}

                # This function defines our RNN model structure
	def neural_network_model(self,data):
		data = tf.transpose(data, [1,0,2])  # This does not have much significance to it
                                                                                       # its just changing and reforming the shape so that
                                                                                        # tensorflow model will be satified with it !
		data = tf.reshape(data,[-1,self.chunk_size])
		data = tf.split(0,self.n_chunks, data)
		lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)   # The LSTM Cell function in TensorFlow
		outputs, states = rnn.rnn(lstm_cell, data, dtype=tf.float32)

		# (input_data * weights) + biases
		output = tf.add(tf.matmul(outputs[-1],self.layer['weights']),self.layer['biases'])
		return output
		
                # This is a function that tests how RNN classifier works
                # in training phase for our training data set
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
					batch_x, batch_y = my_data.images_to_square_data_batch(batch_number,self.batchsize,self.reshape_size)
					batch_x = batch_x.reshape((self.batchsize*self.number_of_classes,self.n_chunks,self.chunk_size))
					_, c = sess.run([optimizer, cost], feed_dict = {self.x: batch_x, self.y: batch_y})
					epoch_loss += c
				print('Epoch', epoch, 'completed out of', self.epochs,' loss:',epoch_loss)

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))

			# validation stage
			test_data,test_label = my_data.get_square_test_set(self.reshape_size)
			test_data = test_data.reshape(-1,self.n_chunks,self.chunk_size)
			print('Accuracy:',accuracy.eval({self.x:test_data, self.y:test_label}))

			# save model value
			saver.save(sess, 'model/RNN_model.chk')
		sess.close()
		
                # This is a function that tests how RNN classifier works
                # in testing phase for our testing data set
	def test_neural_network(self, my_data):
		print "testing phase"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare saver object
		saver = tf.train.Saver(tf.all_variables())
				
		with tf.Session() as sess:
			saver.restore(sess,'model/RNN_model.chk')

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			
			# validation stage
			test_data,test_label = my_data.get_square_test_set(self.reshape_size)
			test_data = test_data.reshape(-1,self.n_chunks,self.chunk_size)
			print('Accuracy:',accuracy.eval({self.x:test_data, self.y:test_label}))
		sess.close()

                # This is a function that tests how RNN classifier works
                # for identifying surgical instruments in real time surgical operations video
	def real_world_neural_network(self, my_data, scale):
		print "real world implementation"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare neural network object
		saver = tf.train.Saver(tf.all_variables())
		
		with tf.Session() as sess:
			saver.restore(sess,'model/RNN_model.chk')

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			predict = tf.argmax(prediction,1)
			
			# validation stage
			while my_data.cap.isOpened():

				# prepare video data object
				ret, src = my_data.cap.read() # faster!
				ret, src = my_data.cap.read()
				frame = my_data.resize(src,scale)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# extract input data for neural network
				_,test_data = my_data.patchify(gray,True,self.reshape_size) # image,square,reshape_size		
				test_data = test_data.reshape(-1,self.n_chunks,self.chunk_size)

				# obtain neural network output							
				fire_index = predict.eval(feed_dict={self.x: test_data}, session=sess)

				# draw it on the gray image
				gray = my_data.draw_result(gray, fire_index)
				
				# display classification result
				cv2.imshow('frame',frame)
			    	cv2.imshow('gray',gray)
			    	if cv2.waitKey(1) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
		sess.close()
		cap.release()
		cv2.destroyAllWindows()



# Class/CNN_net
# breif/ onvolutional neural network
# desc/ This is a class for defining our CNN network
#           which saves/creates the structure of the model
#           define the training procedure, testing procedure
#           and real world implementation procedure
#           There are more complex computation in CNN, so we also created some helper functions
#           like convolution and so on...
class CNN_net:
                # This is the initialization of the class objects
	def __init__(self,number_of_samples,epochs,batchsize,n_nodes,sizes,keep_rate):
		self.epochs = epochs
		self.batchsize = batchsize
		self.number_of_samples = number_of_samples

		self.n_features = [n_nodes[0],n_nodes[1],n_nodes[2]]
		self.fc_nodes = n_nodes[3] 		
		self.number_of_classes = n_nodes[4]
		self.window_size = sizes[0]
		self.fc_imgsize = sizes[1]   # fully connected layer
		self.reshape_size = sizes[2]
		
		self.x = tf.placeholder('float',[ None, self.reshape_size*self.reshape_size]) 
		self.y = tf.placeholder('float') 

		self.keep_rate = keep_rate
		self.keep_prob = tf.placeholder('float') # mimic dead neuron

		randnorm1 = tf.random_normal([self.window_size, self.window_size, self.n_features[0], self.n_features[1]])
		randnorm2 = tf.random_normal([self.window_size, self.window_size, self.n_features[1], self.n_features[2]])
		randnorm3 = tf.random_normal([self.fc_imgsize*self.fc_imgsize*self.n_features[2],self.fc_nodes])
		randnorm4 = tf.random_normal([self.fc_nodes,self.number_of_classes])

		self.weights = {'conv1':  tf.Variable(randnorm1,name = 'W_conv1'),
				'conv2':  tf.Variable(randnorm2,name = 'W_conv2'),
				'fc':     tf.Variable(randnorm3,name = 'W_fc'), 
				'output': tf.Variable(randnorm4,name = 'W_output')
		}
		self.biases = { 'conv1':  tf.Variable(tf.random_normal([self.n_features[1]]),name = 'b_conv1'),
				'conv2':  tf.Variable(tf.random_normal([self.n_features[2]]),name = 'b_conv1'),
				'fc':     tf.Variable(tf.random_normal([self.fc_nodes]), name = 'b_fc'),
				'output': tf.Variable(tf.random_normal([self.number_of_classes]),name = 'b_output')
		}
	
	def conv2d(self,x, W):
		# no depth in here
		return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME') # fights against local minimum

	def maxpool2d(self,x):
		# size of window movement of window
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

                # This function defines our CNN model structure
	def neural_network_model(self,data):
		data = tf.reshape(data, shape=[-1,self.reshape_size,self.reshape_size,1])
		conv1 = tf.nn.relu(self.conv2d(data, self.weights['conv1'])+self.biases['conv1'])
		conv1 = self.maxpool2d(conv1)
		conv2 = tf.nn.relu(self.conv2d(conv1, self.weights['conv2'])+self.biases['conv2'])
		conv2 = self.maxpool2d(conv2)
		fc = tf.reshape(conv2,[-1, self.fc_imgsize*self.fc_imgsize*self.n_features[2]])
		fc = tf.nn.relu(tf.matmul(fc,self.weights['fc'])+self.biases['fc'])
		fc = tf.nn.dropout(fc,self.keep_rate)
		output = tf.matmul(fc,self.weights['output'])+self.biases['output']
		return output
		
                # This is a function that tests how CNN classifier works
                # in training phase for our training data set
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
					batch_x, batch_y = my_data.images_to_square_data_batch(batch_number,self.batchsize,self.reshape_size)
					_, c = sess.run([optimizer, cost], feed_dict = {self.x: batch_x, self.y: batch_y})
					epoch_loss += c
				print('Epoch', epoch, 'completed out of', self.epochs,' loss:',epoch_loss)

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))

			# validation stage
			test_data,test_label = my_data.get_square_test_set(self.reshape_size)
			print('Accuracy:',accuracy.eval({self.x:test_data, self.y:test_label}))

			# save model value
			saver.save(sess, 'model/CNN_model.chk')
		sess.close()

                # This is a function that tests how CNN classifier works
                # in testing phase for our testing data set
	def test_neural_network(self, my_data):
		print "testing phase"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare saver object
		saver = tf.train.Saver([self.weights['conv1'],self.weights['conv2'],self.weights['fc'],self.weights['output'],self.biases['conv1'],self.biases['conv2'],self.biases['fc'],self.biases['output']])
				
		with tf.Session() as sess:
			saver.restore(sess,'model/CNN_model.chk')

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			
			# validation stage
			test_data,test_label = my_data.get_square_test_set(self.reshape_size)
			print('Accuracy:',accuracy.eval({self.x:test_data, self.y:test_label}))
		sess.close()


                # This is a function that tests how CNN classifier works
                # for identifying surgical instruments in real time surgical operations video
	def real_world_neural_network(self, my_data, scale):
		print "real world implementation"
		# prepare model 
		prediction = self.neural_network_model(self.x)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		# prepare neural network object
		saver = tf.train.Saver([self.weights['conv1'],self.weights['conv2'],self.weights['fc'],self.weights['output'],self.biases['conv1'],self.biases['conv2'],self.biases['fc'],self.biases['output']])
		
		with tf.Session() as sess:
			saver.restore(sess,'model/CNN_model.chk')

			correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			predict = tf.argmax(prediction,1)
			
			# validation stage
			while my_data.cap.isOpened():

				# prepare video data object
				ret, src = my_data.cap.read() # faster!
				ret, src = my_data.cap.read()
				frame = my_data.resize(src,scale)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# extract input data for neural network
				_,test_data = my_data.patchify(gray,True,self.reshape_size) # image,square,reshape_size		
	
				# obtain neural network output							
				fire_index = predict.eval(feed_dict={self.x: test_data}, session=sess)

				# draw it on the gray image
				gray = my_data.draw_result(gray, fire_index)
				
				# display classification result
				cv2.imshow('frame',frame)
			    	cv2.imshow('gray',gray)
			    	if cv2.waitKey(1) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
		sess.close()
		cap.release()
		cv2.destroyAllWindows()
