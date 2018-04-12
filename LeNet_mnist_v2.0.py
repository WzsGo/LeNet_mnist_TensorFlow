#encoding:utf-8
"""
DtatSet : Mnist
Network model : LeNet-4
Order:
    1.Accuracy up to 98%
    2.Output result of True and Test labels
    3.Output image and label
    4.ues tf.layer.**  funcation
Time : 2018/04/10
Author:zswang

"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#define parameteer
image_size = 784    #28*28+784
out_class = 10
display_step = 200

#define super parameter
train_keep_prop = 0.5
batch_size = 100
epcoh = 1600

#read dataset and set vector as one_hot
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)  

#read test dataset
test_x = mnist.test.images[:1000]
test_y = mnist.test.labels[:1000]

#define placeholder
xs = tf.placeholder(tf.float32,[None,image_size])   
ys = tf.placeholder(tf.float32,[None,out_class])  
keep_prob = tf.placeholder(tf.float32)     
#reshape image to vector [samples_num,28,28,1]
x_image = tf.reshape(xs,[-1,28,28,1])       #-1:all of train dataset images        
                                           
#crate Lenet-4 
conv1 = tf.layers.conv2d(inputs=x_image,filters=32,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu) 
                                                                    #filters = output channel
                                                                    #conv-1 [28,28,1]->>[28,28,32]
                                                                   
pool1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)   #pool_size ：kenel_size
                                                                    #pool-1 [28,28,32]->>[14,14,32] 
conv1_drop = tf.nn.dropout(pool1,keep_prob)                                                                    


conv2 = tf.layers.conv2d(conv1_drop,64, 5, 1, 'same', activation=tf.nn.relu) 
                                                                    #conv-2 [14,14,32]->>[14,14,64]

pool2 = tf.layers.max_pooling2d(conv2, 2, 2)                   #pool-2 [14,14,64]->>[7,7,64]

conv2_drop = tf.nn.dropout(pool2,keep_prob)


flat = tf.reshape(conv2_drop, [-1, 7*7*64])                              #flaten [7,7,64]->>[7*7*64]

pre_fcn1 = tf.layers.dense(flat, 1024)                              #fucn-1 [7*7*64]->>[1024]

prediction = tf.layers.dense(pre_fcn1, 10)                          #fucn-2 [1024]->>[10]
#computer loss,As target funcation
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)  
#define gradent dencent model to minimize loss(target funcation)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#computer accuracy
accury_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1)),tf.float32))
                                                        #The compare object :prediction and ys(train or test) 
                                                        #tf.cast() : change data dtype
                                                        #tf.argmax() : output the index of maxnum in array

init = tf.global_variables_initializer()                #init parametre step1                 
with tf.Session() as sess:
    sess.run(init)                                      #init parametre step2
    for i in range(epcoh):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size) 
        sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:1})  
        if i % display_step == 0:
            print("------------------")
            #print('train loss : '?+str(sess.run(loss,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:1})))
            print('Train Accuracy : '+str(sess.run(accury_train,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:train_keep_prop})))
            print('Test Accuracy : '+str(sess.run(accury_train,feed_dict = {xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1})))                                                       
    for j in range(1):
        print("--------------------Compare to True and Test----------------------------")
        plt.imshow(test_x[j].reshape((28,28)), cmap='gray') 
        plt.show()
        print("True label ："+str(np.argmax(test_y[0:j+1],1)))
        pre_prop = sess.run(prediction,{xs:test_x[0:j+1],keep_prob:1})
        print("Test label ："+str(np.argmax(pre_prop,1)))
      


