import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#使用Xaiver initialization进行参数初始化，实现标准的均匀分布

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

#定义一个去噪自编码的class

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input, n_hidden, transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                            self.x + scale * tf.random_normal((n_input,)),
                            self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']),self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.substract(
                            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

     #参数初始化函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(zeros([self.n_hidden],dtype = tf.float32))
        all_weights['w2'] = tf.Variable(zeros([self.n_hidden,self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(zeros([self.n_input], dtype=tf.float32))
        return all_weights

    #计算损失及进一步训练的函数
    def pertial_fit(self,X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost

    #只求损失的函数
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})

    #返回自编码器隐含层的输出结果
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})

    #将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
         return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

    #定义reconstruct函数，整体运行一次复原过程，包括提取高阶特征和通过高阶特征复原数据，
    #即包括transform和generate两个方面
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale:self.training_scale})

    def getWeights(self):#获得隐含层的权重w1
        return self.sess.run(self.weights['w1'])

    def getBiases(self):#获得隐含层的偏置系数b1
        return self.sess.run(self.weights['b1'])

#载入mnist数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#对训练、测试数据进行标准化处理
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

#定义一个获取随机block数据的函数
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)   #总训练样本数
training_epochs = 20    #最大训练轮数
batch_size = 128
display_step = 1    #每隔一轮显示一次损失cost

#创建一个AGN自编码器实例，定义模型输入节点数n_input为784，自编码器的隐含层节点数为200，隐含层的激活函数为
#softplus，优化器为Adam且学习速率为0.001，将噪声的系数scale设为0.01

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                               n_hidden = 200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                               scale = 0.01)


#下面开始训练过程，
#每一轮循环开始时，将平均损失avg_cost设为0，并计算总共需要的batch数（样本总数除以batch大小），
#在每一个batch循环中，先使用get_random_from_data函数随机抽取一个block的数据，然后使用成员函数partial_fit
#训练这个batch的数据并计算当前的cost，最后将当前的cost整合avg_cost中。

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=",
              "{:.9f}".format(avg_cost))


print("total cost: " + str(autoencoder.calc_total_cost(X_test)))

