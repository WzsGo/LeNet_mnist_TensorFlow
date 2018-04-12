#encoding:utf-8
"""
DtatSet : Mnist
Network model : LeNet-4
Order:
    1.Accuracy up to 97%
    2.Output result of True and Test labels
    3.Output image and label
    4.ues tf.layer.**  funcation
Time : 2018/04/10
Author:zswang

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#define parameteer
image_size = 784    #28*28=784
out_class = 10
display_step = 200

#define super parameter
train_keep_prop = 0.5
batch_size = 100
epcoh = 1000

#read data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)  #one-hot向量表示只有一个数值是1

#define placeholder
xs = tf.placeholder(tf.float32,[None,image_size]) 
ys = tf.placeholder(tf.float32,[None,out_class])  
keep_prob = tf.placeholder(tf.float32)     
##reshape image to vector [samples_num,28,28,1]
x_image = tf.reshape(xs,[-1,28,28,1])      #-1:all of train dataset images
                                           
#define weight and bias 
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#define conve
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')                       #trides[0] = strides[3] = 1
#define maxpooling
def max_pool_2d(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')   #trides[0] = strides[3] = 1

                                                        #flaten  [28,28,1] ->> [28,28,32]
W_con1 =  weight_variable([5,5,1,32])                   #kenel:[5,5],in_size = 1 out_size = 32
b_con1 =  bias_variable([32])                           
h_conv1 = tf.nn.relu(conv2d(x_image,W_con1) + b_con1)   #input data = x_image
h_conv1_drop = tf.nn.dropout(h_conv1,keep_prob)
                                                        #pool-1  [28,28,32] ->> [14,14,32]
h_pool1 = max_pool_2d(h_conv1_drop)                     
                                                        
                                                        #conv-2  [14,14,32] ->> [14,14,64]
W_con2 = weight_variable([5,5,32,64])                   #kenel:[5,5],in_size = 32,out_size = 64
b_con2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_con2) + b_con2)  
h_conv2_drop = tf.nn.dropout(h_conv2,keep_prob)
                                                        #conv-2  [14,14,64] ->> [7,7,64]
h_pool2 = max_pool_2d(h_conv2_drop)                     

                                                        #fucn-1  [7,7,64] ->> [7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])          
                                                        
                                                        #fucn-1  [7*7*64] ->> [1024]
W_fc1 = weight_variable([7*7*64,1024])                  
b_fc1 = bias_variable([1024])                                              
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)   
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)             

                                                        #fucn-2  [1024]->>[10]
W_fc2 = weight_variable([1024,10])                      
b_fc2 = bias_variable([10])                             
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2) 

#computer loss,As target funcation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
#define gradent dencent model to minimize loss(target funcation)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#computer accuracy
accury_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1)),tf.float32))
                                                #The compare object :prediction and ys(train or test) 
                                                #tf.cast() : change data dtype
                                                #tf.argmax() : output the index of maxnum in array         

init = tf.global_variables_initializer()                             
with tf.Session() as sess:
    sess.run(init)                                      
    for i in range(epcoh):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size) 
        sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:1})  
        if i % display_step == 0: 
            #print('Train loss : '+str(sess.run(cross_entropy,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:1})))
            print('train accuracy:'+str(sess.run(accury_train,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:train_keep_prop})))
            print('test accuracy:'+str(sess.run(accury_train,feed_dict = {xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1})))
            print("------------------")
    for j in range(1):
        print("--------------------Compare to True and Test----------------------------")
        plt.imshow(test_x[j].reshape((28,28)), cmap='gray') 
        plt.show()
        print("True label ："+str(np.argmax(test_y[0:j+1],1)))
        pre_prop = sess.run(prediction,{xs:test_x[0:j+1],keep_prob:1})
        print("Test label ："+str(np.argmax(pre_prop,1)))
                                                      