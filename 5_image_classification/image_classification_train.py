"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, [None, image_shape[0],image_shape[1],image_shape[2]], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution (4,4)
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function (?, 784) (?, 28, 28, 1)
# (?, 784)
# (?, 28, 28, 1)
# (?, 14, 14, 32)
# (?, 7, 7, 64)
# (?, 1024)
# (?, 10)

    # 卷积核 [conv_ksize[0]*[conv_ksize[1]，出入数，输出数
    
    layer1 = tf.nn.conv2d(x_tensor, tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], int(x_tensor.shape[3]), conv_num_outputs], stddev=0.1)), strides=[1, conv_strides[0], conv_strides[1], 1], padding="SAME")
    layer1 = tf.nn.bias_add(layer1, tf.Variable(tf.random_normal([conv_num_outputs], stddev=0)))
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.nn.max_pool(layer1, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding="SAME")

#     out = tf.add(tf.matmul(layer2, tf.Variable(tf.random_normal([1024, conv_num_outputs]))), tf.Variable(tf.random_normal([n_classes])))

    return layer2 # [None, 4, 4, 10]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function

    fc1 = tf.reshape(x_tensor, [-1, int(x_tensor.shape[1]*x_tensor.shape[2]*x_tensor.shape[3])])

    return fc1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    fc1 = tf.add(tf.matmul(x_tensor, tf.Variable(tf.random_normal([int(x_tensor.shape[1]), num_outputs], stddev=0.1))), tf.Variable(tf.random_normal([num_outputs], stddev=0)))
    fc1 = tf.nn.relu(fc1)
    return fc1 # [None, 40]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
#     print(x_tensor.shape)
    out = tf.add(tf.matmul(x_tensor, tf.Variable(tf.random_normal([int(x_tensor.shape[1]), num_outputs], stddev=0.1))), tf.Variable(tf.random_normal([num_outputs], stddev=0)))
#     print(out.shape)
    
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

    print(x.shape)
    con_1 = conv2d_maxpool(x, 16, (3, 3), (1, 1), (3, 3), (2, 2))
    print(con_1.shape)
    con_1 = conv2d_maxpool(con_1, 32, (3, 3), (1, 1), (3, 3), (2, 2))
    print(con_1.shape)
    con_1 = conv2d_maxpool(con_1, 64, (3, 3), (1, 1), (3, 3), (2, 2))
    print(con_1.shape)
    
    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flat_1 = flatten(con_1)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    full_1 = fully_conn(flat_1, 64)
    full_1 = tf.nn.dropout(full_1, keep_prob)
    full_1 = fully_conn(full_1, 32)
    full_1 = tf.nn.dropout(full_1, keep_prob)
#     full_1 = fully_conn(full_1, 16)
#     full_1 = tf.nn.dropout(full_1, keep_prob)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(full_1, 10)
    
    # TODO: return output
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
        
    pass


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss, _ = session.run([cost, accuracy], feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    _, acc = session.run([cost, accuracy], feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})
    print("Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    pass


# TODO: Tune Parameters
epochs = 20
batch_size = 128
keep_probability = 0.8

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# print('Checking the Training on a Single Batch...')
# with tf.Session() as sess:
#     # Initializing the variables
#     sess.run(tf.global_variables_initializer())
#      
#     # Training cycle
#     for epoch in range(epochs):
#         batch_i = 1
#         for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
#             train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
#         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#         print_stats(sess, batch_features, batch_labels, cost, accuracy)


save_model_path = './image_classification'
 
print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
     
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
             
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
