#下载PTB（Penn Tree Bank）数据集并解压
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xvf simple-examples.tgz

git clone https://github.com/tensorflow/models.git
cd models/tutorials/rnn/ptb
import time
import numpy as np
import tensorflow as tf
import reader

#下面定义语言模型处理输入数据的class，
class PTBInput(object):
    def __init__(self, config, data, name = None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name = name
        )

#定义语言模型的class，PTBModel
class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self.input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

#使用tf.contrib.rnn.BasicLSTMCell设置默认的LSTM单元
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias = 0.0, state_is_tuple=True
            )
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob = config.keep_prob
                )
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)],
            state_is_tuple=True
        )
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        #创建网络的词嵌入embedding部分，即将one-hot编码格式的单词转化为向量表达形式
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype = tf.float32
            )
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        #定义输出outputs
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(inputs[:, time_step, :],state)
                #inputs有3个维度，第一个维度代表是batch中的第几个样本，第二个维度代表是样本中的第几个单词
                #第三个维度是单词的向量表达的维度,而inputs[:, time_step, :]代表所有样本的第time_step个单词
                outputs.append(cell_output)

        #将output的内容用tf.concat串接到一起，并用tf.reshape将其转为一个很长的一位向量
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size,vocab_size], dtype = tf.float32
        )
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype = tf.float32)]
        )
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        #定义学习速率的变量_lr,并将其设为不可训练
        self._lr = tf.Variable(0.0, trainable = False)
        tvars = tf.trainable_variables()
        #利用下式设置梯度的最大范数max_grad_norm
        grabs, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step = tf.contrib.framework.get_or_creat_global_step())

        #设置一个名为_new_lr(new learning rate)的placeholder用以控制学习速率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name = "new_learning_rate"
        )
        self._lr_update = tf.assign(self._lr, self._new_lr)
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})

        #python中的@property装饰器可以讲返回变量设为只读，防止修改变量引发的问题
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

#接下来定义几种不同大小的模型的参数
class SmallConfig(object):
    init_scale = 0.1    #网络初始权重的scale
    learning_rate = 1.0 #学习速率的初始值
    max_grad_norm = 5   #梯度的最大范数
    num_layers = 2      #LSTM可以堆叠的层数
    num_steps = 20      #LSTM反向传播的展开步数
    hidden_size = 200   #LSTM内的隐含节点数
    max_epoch = 4       #初始学习速率可训练的epoch数
    max_max_epoch  = 13 #总共可训练的epoch数
    keep_prob  = 1.0    #dropout层保留节点的比例
    lr_decay = 0.5      #学习速率的衰减速度
    batch_size = 20     #每个batch中的样本数量
    vocab_size = 10000  #词汇表大小


class MediumConfig(object):
    #减小了init_scale，即希望权重初值不要太大，小一些有利于温和的训练
    #num_steps增大到35，hidden_size和max_max_epoch也相应增大约3倍
    #因为学习的迭代次数增大，因此学习速率的衰减速率减小了
    init_scale = 0.05    #网络初始权重的scale
    learning_rate = 1.0 #学习速率的初始值
    max_grad_norm = 5   #梯度的最大范数
    num_layers = 2      #LSTM可以堆叠的层数
    num_steps = 35      #LSTM反向传播的展开步数
    hidden_size = 650   #LSTM内的隐含节点数
    max_epoch = 6       #初始学习速率可训练的epoch数
    max_max_epoch  = 39 #总共可训练的epoch数
    keep_prob  = 0.5    #dropout层保留节点的比例
    lr_decay = 0.8      #学习速率的衰减速度
    batch_size = 20     #每个batch中的样本数量
    vocab_size = 10000  #词汇表大小

class LargeConfig(object):
    #进一步缩小init_scale，放宽max_grad_norm
    init_scale = 0.04    #网络初始权重的scale
    learning_rate = 1.0 #学习速率的初始值
    max_grad_norm = 10   #梯度的最大范数
    num_layers = 2      #LSTM可以堆叠的层数
    num_steps = 35      #LSTM反向传播的展开步数
    hidden_size = 1500   #LSTM内的隐含节点数
    max_epoch = 14       #初始学习速率可训练的epoch数
    max_max_epoch  = 55 #总共可训练的epoch数
    keep_prob  = 0.35    #dropout层保留节点的比例
    lr_decay = 1/1.15    #学习速率的衰减速度
    batch_size = 20     #每个batch中的样本数量
    vocab_size = 10000  #词汇表大小

class TestConfig(object):
    #只是作为测试用，参数都尽量使用最小值
    init_scale = 0.1    #网络初始权重的scale
    learning_rate = 1.0 #学习速率的初始值
    max_grad_norm = 1   #梯度的最大范数
    num_layers = 1      #LSTM可以堆叠的层数
    num_steps = 2      #LSTM反向传播的展开步数
    hidden_size = 2   #LSTM内的隐含节点数
    max_epoch = 1       #初始学习速率可训练的epoch数
    max_max_epoch  = 1 #总共可训练的epoch数
    keep_prob  = 1.0    #dropout层保留节点的比例
    lr_decay = 0.5    #学习速率的衰减速度
    batch_size = 20     #每个batch中的样本数量
    vocab_size = 10000  #词汇表大小

#下面定义训练一个epoch数据的函数run_eopch
def run_epoch(session, model, eval_op = None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(cost / iters)

raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config,data=train_data,name = "Traininput")
        with tf.variable_scope("model", reuse=None,initializer=initializer):
            m = PTBModel(is_training=True,config=config,input_=train_input)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config,data=valid_data,name="ValidInput")

        with tf.variable_scope("Model",reuse= True,initializer=initializer):
            mvalid = PTBModel(is_training=False,config=config,input_=valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config,data=test_data,name="TestInput")
        with tf.variable_scope("Model",reuse=True,initializer=initializer):
            mtest = PTBModel(is_training=False,config=eval_config,input_=test_input)

    #使用tf.train.Supervisor()创建训练的管理器sv
    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op,verbose = True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, valid_perplexity))
        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)

#在图像标题生成任务中，包含注意力机制的RNN可以对某一区域的图像进行分析，并生成对应的文字描述
#可阅读论文Show,Attend and Tell: Neural Image Caption Generation with Visual Attention