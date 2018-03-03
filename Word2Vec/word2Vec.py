import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

#先定义下载文本数据的函数，这里使用urllib.request.urlretrieve下载数据的压缩文件并核对文件尺寸
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raie Exception(
            'Failed to verify' + filename + '. Can you get to it with a browser:'
        )
    return filename

filename = maybe_download('text8.zip', 31344016)

#解压下载文件
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compact.as_str(f.read(f.namelist()[0])).split()
        #将数据转成单词的列表
    return data

words = read_data(filename)
print('Data size', len(words))

#创建vocabulary词汇表
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    #统计单词列表中单词的频数，取top 50000频数的单词作为vocabulary
    dictionary = dict()
    #创建一个dict，把top 50000频数的单词作为vocabulary放入dictionary中

    #下面遍历单词列表，对其中每一个单词，先判断是否出现在dictionary中，是则转为其编号，
    # 不是则转为编号0（Unknow）。最后返回转换后的编码（data）、
    #每个单词的频数统计（count）、词汇表（dictionary）及其反转的形式（reverse_dictionary）
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            indx = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)


#删除原始单词列表，节约内存
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

#下面生成Word2Vec的训练样本
data_index = 0
#用来生成训练用的batch数据，batch_size为batch的大小，skip_window为单词最远可以联系的距离
#nums_skips为对每个单词生成多少个样本，不能大于skip_window的两倍，且batch_size必须为他的整数倍
def generate_batch(batch_size, num_skips, skip_window):
    global data_index    #定义其为global变量，要确保data_index在generate_batch中可以被修改
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarry(shape=(batch_size), dtype = np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1 #定义为对某个单词创建相关样本时会使用到的单词数量，包括单词本身和他前后的单词
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.appand(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

#调用generate_batch函数测试其功能
batch, labels = generate_batch(batch_size = 8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i,0],reverse_dictionary[labels[i,0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
    train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,embedding_size], -1.0, 1.0)
        )
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                             stddev = 1.0 / math.sqrt(embedding_size))
    )
        nce_biasses = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                     biases = nce_biases,
                                     labels = train_labels,
                                     inputs = embed,
                                     num_sampled = num-sampled,
                                     num_classes = vocabulary_size))

    #定义优化器为SGD，学习速率为1.0
    optimizer= tf.train.gradirntDescentOptimizer(1.0).minisize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.sqare(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

    init = tf.global_variables_initializer()

num_steps = 10001

with tf.Session(graph = graph) as session:
    init.run()
    print("initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
        average_loss += loss_val

        #之后每2000次循环，计算一下平均loss并显示出来
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ":", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = "%s %s," %(log_str, close_word)
            print(log_str)
            final_embeddings = normalized_embeddings.eval()

#定义一个用来可视化Word2Vec效果的函数
def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize = (18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy = (x,y),
                     xytext = (5,2),
                     textcoords = 'offset points',
                     ha = 'right',
                     va = 'bottom')

    plt.savefig(filename)

#使用sklearn.manifold.TSNE实现降维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity = 30, n_components = 2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)