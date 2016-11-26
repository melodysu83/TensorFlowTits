from inputdata import Dataset

"""input > weight > hidden L1 (activation function) > weights > hidden L2
(activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)

back propagation

feed forward + backprop = epoch
"""

# Prepare data
paths = ["Desktop/TensorFlowTuts/tooldata/","Desktop/TensorFlowTuts/nontooldata/"]
number_of_samples = 38000
number_of_classes = 2
image_width = 30
image_height = 40
image_type = "tiff"

SurgicalData = Dataset(paths,number_of_samples,number_of_classes,image_width, image_height, image_type)


# design neural network parameters
batch_size = 100
n_nodes = [  1200,      256,      256,     10]
#           input, hiddenL1, hiddenL2, output
#                          [height,     width]
x = tf.placeholder('float',[  None,n_nodes[0]])  # our input data (28*28=784 pixels per data)
y = tf.placeholder('float')              # the label of our data

# construct neural network
def neural_network_model(data):
	weights = {
		'h1': tf.Variable(tf.random_normal([n_nodes[0], n_nodes[1]])),
		'h2': tf.Variable(tf.random_normal([n_nodes[1], n_nodes[2]])),
		'output': tf.Variable(tf.random_normal([n_nodes[2], n_nodes[3]]))
	}
	biases = {
		'h1': tf.Variable(tf.random_normal([n_nodes[1]])),
		'h2': tf.Variable(tf.random_normal([n_nodes[2]])),
		'output': tf.Variable(tf.random_normal([n_nodes[3]]))
	} 

	# (input_data * weights) + biases
	n_layer1 = tf.add(tf.matmul(data,weights['h1']),biases['h1'])
	n_layer1 = tf.nn.relu(n_layer1)
	n_layer2 = tf.add(tf.matmul(n_layer1,weights['h2']),biases['h2'])
	n_layer2 = tf.nn.relu(n_layer2)
	output = tf.add(tf.matmul(n_layer2,weights['output']),biases['output'])
	return output

def train_neural_network(x,total_epochs,batch_size):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(total_epochs):
			epoch_loss = 0
			for batch_number in range(int(number_of_samples/batch_size)):
				batch_x, batch_y = SurgicalData.images_to_data_batch(batch_number,batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs,' loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))

		test_data,test_label = SurgicalData.get_test_set()
		print('Accuracy:',accuracy.eval({x:test_data, y:test_label}))


epochs = 15
batchsize = 100
train_neural_network(x,epochs,batchsize)
