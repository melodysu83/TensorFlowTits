import copy
import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell


# load training data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) 

# design neural network parameters
hm_epochs = 10
batch_size = 128

reshape_size = 28
window_size = 5
n_features0 = 1
n_features1 = 32
n_features2 = 64
fc_imgsize = 7
fc_nodes = 1024
n_classes = 10  # output dimension

x = tf.placeholder('float',[  None, 784])  # our input data (28*28=784 pixels per data)
y = tf.placeholder('float')              # the label of our data

keep_rate = 0.8 
keep_prob = tf.placeholder(tf.float32) # mimic dead neuron
                                       # fights against local minimun

def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME') # no depth in here

def maxpool2d(x):
	#                        size of window   movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# construct neural network
def convolutional_neural_network(x):
	weights = {'W_conv1':tf.Variable(tf.random_normal([window_size, window_size, n_features0, n_features1])),
	           'W_conv2':tf.Variable(tf.random_normal([window_size, window_size, n_features1, n_features2])),
	           'W_fc':tf.Variable(tf.random_normal([fc_imgsize*fc_imgsize*n_features2,fc_nodes])), # fully connected layer
	           'output': tf.Variable(tf.random_normal([fc_nodes,n_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([n_features1])),
	           'b_conv2':tf.Variable(tf.random_normal([n_features2])),
	           'b_fc':tf.Variable(tf.random_normal([fc_nodes])),
	           'output': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, shape=[-1,reshape_size,reshape_size,1])

	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+biases['b_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])+biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2,[-1, fc_imgsize*fc_imgsize*n_features2])
	fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

	fc = tf.nn.dropout(fc,keep_rate)

	output = tf.matmul(fc,weights['output'])+biases['output']

	return output

def train_neural_network(x):
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs,' loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))

		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
