import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#设置训练参数
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

n_input = 28        #图像的宽
n_steps = 28        #图像的高
n_hidden = 256      #LSTM的隐藏节点数
n_classes = 10      #MNIST数据集的分类数目

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))
#因为是双向LSTM，有forward和backward 两个LSTm的cell，所以weights的参数量也翻倍，变为2*n_hidden

#定义BidirectionalLSTM网络的生成函数
def BiRNN(x, weights, biases):

    x = tf.transpose(x, [1,0,2])        #把第一个维度batch_size和第二个维度n_steps进行交换
    x = tf.reshape(x, [-1, n_input])    #将输入x变形为(n_steps*batch_size,n_input)的形状
    x = tf.split(x, n_steps)            #将x拆分为长度为n_steps的列表，列表中每个tensor的尺寸都是（batch_size, n_input）

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                            lstm_bw_cell, x, dtype = tf.float32)
    return tf.matmul(outputs[-1], weights) + biases

#利用刚才定义好的函数生成Bidirectional LSTM网络
pred = BiRNN(x, weights, biases)

#对输出结果使用tf.nn.softmax_cross_entropy_with_logits进行softmax处理并计算损失，然后使用tf.reduce_mean计算平均cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#用tf.argmax得到预测模型类别，然后用tf.equal判断是否预测正确
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#用tf.reduce_mean求得平均准确率

init = tf.global_variables_initializer()

#开始执行训练和测试操作
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: atch-x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            print("Iter" + str(step*batch_size) + ", Minibatch Loss =" + \
                  "{:.6f}".format(loss) + ", Training Accuracy = " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    #全部训练迭代结束后，使用训练好的模型，对mnist.test.images中全部的测试数据进行预测，并将准确率展示出来
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x:test_data, y:test_label}))


