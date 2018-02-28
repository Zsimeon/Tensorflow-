#本程序不涉及实际数据的计算，建立一个完整的AlexNet卷积神经网络，
# 然后对他的每个batch的前馈计算和反馈计算的速度进行测试
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

#定义一个用来显示网络每一层结构的函数print_actications，展示每一个卷积层或池化层输出tensor的尺寸
def print_activations(t):
    print(t.op.name, ' ',t.get_shape().as_list())

#设计AlexNet的网络结构
#定义inference，接受images作为输入，返回最后一层pool5及parameters

def inference(images):
    parameters = []
    # 首先是第一个卷积层conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],
                     dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(images, kernel, [1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [64],dtype = tf.float32),
                             trainable = True,name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name = scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # 目前除了AlexNet，其他经典的卷积神经网络基本都放弃了LRN(主要是效果不明显）
    # 使用LRN也会让前馈、反馈的速度大大下降
    lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize = [1,3,3,1], strides = [1,2,2,1],
                           padding = 'VALID', name = 'pool1')
    print_activations(pool1)


    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],
                                                 dtype = tf.float32,stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [192],
                                         dtype = tf.float32), trainable= True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize = [1,3,3,1], strides = [1,2,2,1],
                           padding = 'VALID', name = 'pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize = [1,3,3,1],strides = [1,2,2,1],
                           padding = 'VALID', name = 'pool5')
    print_activations(pool5)

    return pool5, parameters

#实现一个评估AlexNet每轮计算时间的函数，第一个输入是Session，第二个输入是需要评测的算子，第三个变量时测试的名称
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i %10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(),i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size,3],
                                          dtype = tf.float32,
                                          stddev = 1e-1)
    pool5, parameters = inference(images)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #需要给最后的输出pool5设置一个优化目标loss
    time_tensorflow_run(sess,pool5,"Forward")
    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective, parameters)
    time_tensorflow_run(sess, grad, "Forward-backward")

run_benchmark())