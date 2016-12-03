import cv2
import copy
import random
import tensorflow as tf
import numpy as np

from inputdata import Dataset

"""input > weight > hidden L1 (activation function) > weights > hidden L2
(activation function) > weights > output layer
compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)
back propagation
feed forward + backprop = epoch
"""


# Prepare data
paths = ["tooldata/","nontooldata/"]
epochs = 1
batchsize = 100
number_of_samples = 38000
number_of_classes = 2
image_width = 30
image_height = 40
image_type = "tiff"
print "create dataset"
SurgicalData = Dataset(paths,number_of_samples,number_of_classes,image_width, image_height, image_type)
# design neural network parameters
n_nodes = [ 1200, 256, 256, 64, 2]
# input, hiddenL1, hiddenL2, output
# [height, width]
x = tf.placeholder('float',[ None,n_nodes[0]]) # our input data (28*28=784 pixels per data)
y = tf.placeholder('float') # the label of our data
# construct neural network
def neural_network_model(data):
	weights = {
		'h1': tf.Variable(tf.random_normal([n_nodes[0], n_nodes[1]],name = 'W_h1')),
		'h2': tf.Variable(tf.random_normal([n_nodes[1], n_nodes[2]],name = 'W_h2')),
		'h3': tf.Variable(tf.random_normal([n_nodes[2], n_nodes[3]],name = 'W_h3')),
		'output': tf.Variable(tf.random_normal([n_nodes[3], n_nodes[4]],name = 'W_output'))
	}
	biases = {
		'h1': tf.Variable(tf.random_normal([n_nodes[1]],name = 'b_h1')),
		'h2': tf.Variable(tf.random_normal([n_nodes[2]],name = 'b_h2')),
		'h3': tf.Variable(tf.random_normal([n_nodes[3]],name = 'b_h3')),
		'output': tf.Variable(tf.random_normal([n_nodes[4]],name = 'b_output'))
	}

	# (input_data * weights) + biases
	n_layer1 = tf.add(tf.matmul(data,weights['h1']),biases['h1'])
	n_layer1 = tf.nn.relu(n_layer1)
	n_layer2 = tf.add(tf.matmul(n_layer1,weights['h2']),biases['h2'])
	n_layer2 = tf.nn.relu(n_layer2)
	n_layer3 = tf.add(tf.matmul(n_layer2,weights['h3']),biases['h3'])
	n_layer3 = tf.nn.relu(n_layer3)
	output = tf.add(tf.matmul(n_layer3,weights['output']),biases['output'])
	return output

def train_neural_network(x,total_epochs,batch_size):
	prediction = neural_network_model(x)
	print "softmax setting"
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	print "optimizer setting"
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		print "start session"
		sess.run(tf.initialize_all_variables())
                saver = tf.train.Saver(tf.all_variables())

		for epoch in range(total_epochs):
			epoch_loss = 0
			for batch_number in range(int(number_of_samples/batch_size)):
				
				batch_x, batch_y = SurgicalData.images_to_data_batch(batch_number,batch_size)
				print batch_x
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', epochs,' loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		
		print "evaluate result"
		test_data,test_label = SurgicalData.get_test_set()
		print('Accuracy:',accuracy.eval({x:test_data, y:test_label}))

		print "save model"
		saver.save(sess, 'model/model.chk')

def test_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
                saver = tf.train.Saver(tf.all_variables())
		
		ckpt = tf.train.get_checkpoint_state("model/")
		all_vars = tf.trainable_variables()
		for v in all_vars:
			print(v.name)

		
		# Now you can run the model to get predictions
		#test_data,test_label = SurgicalData.get_test_set()
		#print('Accuracy:',accuracy.eval({x:test_data, y:test_label}))

#print "start the training"
train_neural_network(x,epochs,batchsize)
#test_neural_network(x)
