#下载TensorflowModels库，以便使用其中提供CIFAR-10数据的类
git clone https://github.com/tensorflow/models.git
cd models/tutorials/images/cifar10

#然后载入常用库
import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

#定义batch_size, 训练轮数max_steps, 以及下载CIFAR_10数据的默认路径
max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

#定义初始化weight的函数，使用tf.truncated_normal截断的正态分布来初始化权重
#使用一个L2的loss，用wl控制L2loss的大小，最后得到weight_loss
#把weight loss统一存到一个collection，这个collection命名为“losses”，在后面计算神经网络的总体loss时被用上
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev  =stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

#下面使用cifar10类下载数据集，并解压、展开到默认位置
cifar10.maybe_download_and_extract()

#再使用cifar10_input类中的distorted_input函数产生训练需要使用的数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)

#再使用cifar10_input.inputs函数生成测试数据
images_test, labels_test = cifar10_input.inputs(eval_data = True,
                                                data_dir = data_dir,
                                                batch_size = batch_size)

#创建输入数据的palceholder，包括特征和label
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

#创建第一个卷积层
#5*5卷积核，3个颜色通道，64个卷积核，同时设置weight初始化的标准差为0.05
weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding = 'SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides=[1,2,2,1], padding = 'SAME')
norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

#创建第二个卷积层，上一层的卷积核数量为64，所以本层卷积核尺寸的第三个维度即输入的通道数也需要调整为64；
#还有一个需要注意的地方是这里的bias值全部初始化为0.1，而不是0
#最后调整最大池化层和LRN层的顺序，先进行LRN层的处理，再使用最大池化层
weight2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1,1,1,1], padding = 'SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1],padding = 'SAME')

#在两个卷积层之后，将使用一个全连接层，先把第二层的输出结果flatten，使用tf.reshape函数将样本变成一维向量
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim,384], stddev = 0.04,wl=0.004)
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384,192], stddev = 0.04,wl=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weigh5 = variable_with_weight_loss(shape=[192,10], stddev = 1/192.0,wl=0.0)
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
local5 = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

#计算CNN的loss
def  loss(logits, labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = labels,name = 'cross_entropy_per_example'    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

#接着将logits节点和label_holder传入loss函数获得最终的loss
loss = loss(logits, label_holder)
#优化器选择Adam Optimizer,学习速率设为1e-3
train_op = tf.train.AdamOptimizer(1e-3).minize(loss)
#使用tf.nn.in_top_k函数求输出结果中top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#这一步是启动前面的图片数据增强的线程队列，如果这里不启动线程，那么后续的inference以及训练的操作都是无法开始的
tf.train.strat_queue_runners()

#现在开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],feed_dict = {image_holder:image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss = %.2f (%.1f example/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec,sec_per_batch))

#接下来评测模型在测试集上的准确率
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict = {image_holder:image_batch,
                                                   label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

#最后将准确率的评测结果计算并打印出来
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)





