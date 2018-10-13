# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:12:25 2018

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("G:/mnist/",one_hot = True)
# 正态分布 均值为0，标准差位0.1，最大值为1，最小值为-1
def weights_variable(shape):
    initial= tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#常量 0.1 ，结构为shape的矩阵，shape为其行列
def bais_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#卷积  步长为1，边缘补0
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#池化 步长为1，池化层核函数为2x2，数据缩小4倍
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
#定义输入输出结构
#声明一个占位符，None表示输入数量不定，28*28为图片分辨率
xs=tf.placeholder(tf.float32,[None,28*28])
#类别为0-9，对应输出结果
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
#x_image又把xs reshape成28*28*1的形状，图片为灰色，所以通道为1，作为训练时的input，-1表示输入不定
x_image=tf.reshape(xs,[-1,28,28,1])
#搭建网络 卷积层
#第一层卷积网络
#卷积核为5*5，输入通道为1，输出32个不同的卷积特征图
w_conv1=weights_variable([5,5,1,32])
b_conv1=bais_variable([32])
#卷积操作，输出为28*28*32
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
#池化操作，输出为14*14*32
h_pool1=max_pool_2x2(h_conv1)
#第二层卷积网络 
#32通道卷积，卷积出64个特征图
w_conv2=weights_variable([5,5,32,64])
b_conv2=bais_variable([64])
#卷积操作 输出结果为14*14*64
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
#池化操作 输出结果为7*7*64
h_pool2=max_pool_2x2(h_conv2)
#全连接层操作
w_fc1=weights_variable([7*7*64,1024])
b_fc1=bais_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
#输出层操作
# dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
# 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
w_fc2=weights_variable([1024,10])
b_fc2=bais_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
#定义loss
#计算交叉熵
cross_entropy=-(ys*tf.log(y_conv))
#调用优化器，其实质是为数据使交叉熵最小
train_step=tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
#开始训练及评测
#计算准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(ys,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
#每次拿batch为50的数据进行训练
    for i in range(1000):
        batch=mnist.train.next_batch(50)
        if i %100==0:
            train_accuracy=accuracy.eval(feed_dict={xs:batch[0],ys:batch[1],keep_prob:1.0})
            print ('step %d,training accuracy %g' %(i,train_accuracy))
            train_step.run(feed_dict={xs:batch[0],ys:batch[1],keep_prob:0.5})
    print(accuracy.eval(feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1.0}))
























