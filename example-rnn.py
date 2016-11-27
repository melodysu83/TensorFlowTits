import copy
import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell


# load training data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) 

# design neural network parameters
hm_epochs = 3
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

n_nodes = [784, 10]  #input dimension, output dimension
x = tf.placeholder('float',[  None, n_chunks, chunk_size])  # our input data (28*28=784 pixels per data)
y = tf.placeholder('float')              # the label of our data

# construct neural network
def recurrent_neural_network(x):
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_nodes[1]])),
			 'biases': tf.Variable(tf.random_normal([n_nodes[1]]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x,[-1,chunk_size])
	x = tf.split(0,n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

	# (input_data * weights) + biases
	output = tf.add(tf.matmul(outputs[-1],layer['weights']),layer['biases'])
	return output

def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs,' loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))

		print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape(-1,n_chunks,chunk_size), y:mnist.test.labels}))


train_neural_network(x)
