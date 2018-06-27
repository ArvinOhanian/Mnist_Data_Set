# Mnist_Data_Set
'''
Feed Forward:
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer  

Backpropagation(going back and changing weights):
compare output to intended output > cost/loss function >
optimization function > minimize cost (AdamOptimizer...SGD, AdaGrad, etc.)

feed forward + backpropagation = epoch (1 iteration)
'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#importing the handwritten 0-9 data samples
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)#one_hot means that
#we will be taking only one output from the many outputs of our network

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10#how many classes
batch_size = 100#how big the batches of data are that we run at a time

#making a matrix (height X width)
x = tf.placeholder('float', [None, 784])#our x is the data we are feeding in.
#we are importing 28x28 pixel pictures but we can read them one line at a time,
#hence the height being None/irrelevent and width being 784
y = tf.placeholder('float')#y is the label for our data, what we want the machine to find

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #defining 'weights' as a random number and making its height the amount of pixels
    #coming in per datapoint and its width the amount of nodes in hl1. This will
    #cause each datapoint/pixel to be multiplied by 500 nodes, creating an array [1,n_nodes_hl1]
    #defining 'biases' as a random number the size of our nodes to add to our weights
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #the weights for the next hidden layers are changed to the size that can take in
    #the amount of data from previous hl and biases are adjusted again
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    #the weights here will return a matrix/array [1,n_classes] after being multiplied
    #with previous hl;[1,10] in this case.Biases are applied to new size of output
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #this is just adding all the (data*weights) + biases, l1 stands for layer 1
    l1 = tf.nn.relu(l1)#relu = rectified linear, which is an activation function like a sigmoid
    #this makes the output of each layer equal to something in between 0-1
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)#takes in layer 1 weights and applied layer 2 weights
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)#prediction = what the neural net returns
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    #nn.softmax_cross_entroy_with_logits gives us the amount the net predicted correctly
    #by comparing 'prediction' with y labels. Note that this is not something like 7/10 correct,
    #rather it adds 'distance' of each answer to actual answer
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)#default learning rate = .001
    #optimizer is what makes the corrections in backprop. Uses complicated algorithm
    
    hm_epochs = 10#cycles of feed foward & backprop
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#starts running the code
    
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):#defined batch_size above
                #by dividing num_examples by batch size we know how many batches we need
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)#trains next batch
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                #we define _=optimizer and c=cost and run them. The feed_dict defines values,
                #in this case x&y, that we will feed into the optimizer&cost respectively
                #note that x=features and y=labels
                epoch_loss += c #epoch_loss is the error for the curent epoch.Resets per epoch
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        #tf.argmax returns index of max value.Since we are using one_hot, if the indecies are the
        #same, then we know we predicted correctly. Remember that one_hot defines output values
        #of for example 0,1,2,3 as 0001,0010,0100,1000.
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        #casts all values in the correct list as floats and takes their mean, thereby returning
        #the % accuracy, tf.cast turns boolians into floats like 'True' into 1 or 'false' into 0
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        #runs accuracy function on the test images and evaluates them by comparing to labels

train_neural_network(x)
    
    
    
    
    
    
    
    
    
    


