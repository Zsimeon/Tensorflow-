from datetime import datetime
import math
import time
import tensorflow as tf

#先创建一个函数conv_op，用来创建卷积层并把本层的参数存入参数列表
def conv_op(input_op, name, kh, kw, n_out, dh,dw,p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape = [kh, kw, n_in, n_out], dtype = tf.float32,
                                 initializer = tf.contrib.layers.xavier_initializer_conv2d())

    #使用tf.nn.conv2d对input_op进行卷积处理，步长是dh*dw
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
                            padding = 'SAME')
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z,name = scope)
        p += [kernel,biases]
        return activation

    #下面定义全连接层的创建函数fc_op
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "2",
                                 shape = [n_in, n_out], dtype = tf.float32,
                                 initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out],
                                         dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)
        p += [kernel, biases]
        return activation

    #定义最大池化层的创建函数mpool_op
    #池化尺寸为kh*kw，步长是dh*dw
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides = [1,dh, dw, 1],
                          padding = 'SAME',
                          name = name)

#创建VGGNet-16的网络结构
def inference_op(input_op, keep_prob):
    p = []
#第一段卷积网络，两个卷积层和一个最大池化层
    conv1_1 = conv_op(input_op, name = "conv1_1", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    #第一个卷积层的input_op尺寸为224*224*3，输出尺寸为224*224*64
    conv1_2 = conv_op(conv1_1, name = "conv1_2", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    #第二个卷积层的输入输出尺寸均为224*224*64
    pool1 = mpool_op(conv1_2, name = "pool1", kh = 2, kw = 2, dw = 2, dh = 2)
    #一个标准的2*2的最大池化将输出结果尺寸变为112*112*64

#第二段卷积网络，两个卷积层和一个最大池化层
    conv2_1 = conv_op(pool1, name = "conv2_1", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    conv2_2 = conv_op(conv2_1, name = "conv2_2", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    pool2 = mpool_op(conv2_2, name = "pool2", kh = 2, kw = 2, dw = 2, dh = 2)
    #这一段卷积网络的输出尺寸变为56*56*128

    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)
    #输出通道数增长为256，最大池化层保持不变，因此这一段卷积网络的输出尺寸是28*28*256

    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)
    #输出14*14*512

    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)
    #卷积输出通道数不再增加，输出尺寸7*7*512

    #将第5段卷积网络的输出结果进行扁平化，将每个样本化为长度为7*7*512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name = "resh1")

    #连接一个隐含节点数为4096的全连接层，激活函数为relu，然后连接一个Dropout层，在训练时节点保留率为0.5，预测时为1
    fc6 = fc_op(resh1, name="fc6",  n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name = "fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    #最后连接一个有1000个输出节点的全连接层，并使用Softmax进行处理得到分类输出概率
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p

#评测函数
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict = feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                (datetime.now(), i - num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))

#定义评测主函数
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                              dtype = tf.float32,
                                              stddev = 1e-1))

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images,keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        #通过将keep_prob设为1来执行预测，并使用time_tensorflow_run评测forward的运算时间
        #再计算VGGNet-16的最后全连接层的输出fc8的l2 loss，并使用tf.gradients求相对于这个loss的所有模型参数的梯度
        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward")
        objective= tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Forward-backward")

batch_size = 32
num_batches = 100
run_benchmark()




