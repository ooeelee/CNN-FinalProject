# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:17:37 2017
@author: Joohee Lee
"""
import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv("C:/Users/Jang/Desktop/cnntermp_joohee/fer2013/fer2013/fer2013.csv")
#data.shape
data.head()

np.unique(data["Usage"].values.ravel())
print('The number of training data set is %d'%(len(data[data.Usage == "Training"])))

train_data = data[data.Usage == "Training"]

pixels_values = train_data.pixels.str.split(" ").tolist()
pixels_values = pd.DataFrame(pixels_values, dtype=int)

images = pixels_values.values
images = images.astype(np.float)
images

def show(img):
    show_image = img.reshape(48,48)
    
    plt.imshow(show_image, cmap='gray')
    
#show(images[3])

images = images - images.mean(axis=1).reshape(-1,1)
images = np.multiply(images,100.0/255.0)

each_pixel_mean = images.mean(axis=0)
each_pixel_std = np.std(images, axis=0)

images = np.divide(np.subtract(images,each_pixel_mean), each_pixel_std)
#images.shape
image_pixels = images.shape[1]
print('Flat pixel values is %d'%(image_pixels))

image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)
#image_width

labels_flat = train_data["emotion"].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
print('The number of different facial expressions is %d'%labels_count)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
labels[0]

VALIDATION_SIZE = 1709
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
print ('The number of final training data: %d'%(len(train_images)))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-4)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder('float', shape=[None, image_pixels])
y_ = tf.placeholder('float', shape=[None, labels_count])

W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])
image = tf.reshape(x, [-1,image_width , image_height,1])
#print (image.get_shape()) # =>(27000,48,48,1)

h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
#print (h_conv1.get_shape()) # => (27000,48,48,64)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (27000,24,24,1)
h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, "SAME") + b_conv2)
#print (h_conv2.get_shape()) # => (27000,24,24,128)
h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_pool2 = max_pool_2x2(h_norm2)

def local_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

def local_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

W_fc1 = local_weight_variable([12 * 12 * 128, 3072])
b_fc1 = local_bias_variable([3072])
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


W_fc2 = local_weight_variable([3072, 1536])
b_fc2 = local_bias_variable([1536])
h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])
h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)

keep_prob = tf.placeholder('float')
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1536, labels_count])
b_fc3 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
#print (y.get_shape()) # => (40000, 10)

LEARNING_RATE = 1e-4
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)

TRAINING_ITERATIONS = 3000
DROPOUT = 0.5
BATCH_SIZE = 50

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y_: validation_labels[0:BATCH_SIZE], 
                                                            keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        if i%(display_step*10) == 0 and i and display_step<100:
            display_step *= 10
            
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
    
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools

if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 
                                                   y_: validation_labels, 
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.0, ymin = 0.0)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
    
saver = tf.train.Saver(tf.all_variables())
saver.save(sess, 'C:/Users/Jang/Desktop/cnntermp_joohee/my-model1', global_step=0)

test_data = data[data.Usage == "PublicTest"]
test_data.head()
len(test_data)

test_pixels_values = test_data.pixels.str.split(" ").tolist()
test_pixels_values = pd.DataFrame(test_pixels_values, dtype=int)
test_images = test_pixels_values.values
test_images = test_images.astype(np.float)
test_images = test_images - test_images.mean(axis=1).reshape(-1,1)
test_images = np.multiply(test_images,100.0/255.0)
test_images = np.divide(np.subtract(test_images,each_pixel_mean), each_pixel_std)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

predicted_labels = np.zeros(test_images.shape[0])
N = test_images.shape[0]//BATCH_SIZE

for i in range(N):
#    print('a')
#    predicted_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})
    A = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})
    predicted_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = A
    
print('predicted_labels({0})'.format(len(predicted_labels)))

predicted_labels
test_data.emotion.values

accuracy_score(test_data.emotion.values, predicted_labels)

















