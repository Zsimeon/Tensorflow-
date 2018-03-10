from tensorflow.contrib.distributions import Gamma
import tensorflow as tf
alpha = tf.constant([3.0]*5)
beta = tf.constant(11.0)
gamma = Gamma(alpha=alpha,beta=beta)
shape1 = gamma.batch_shape().eval()
shape2 = gamma.get_batch_shape()

x = np.array([2.5,2.5,4.0,0.1,1.0,2.0],dtype = np.float32)
log_pdf = gamma.log_pdf(x)

x = np.array([2.5,2.5,4.0,0.1,1.0,2.0],dtype = np.float32)
log_pdf = gamma.log_pdf(x)

batch_size = 6
alpha = tf.constant([[2.0,4.0]]*batch_size)
veta=tf.constant([[3.0,4.0]]*batch_size)
x = np.array([[2.5,2.5,4.0,0.1,1.0,2.0]],dtype = np.float32).T
gamma = Gamma(alpha=alpha,beta=beta)
log_pdf = gamma.log_pdf(x)


###############################
#Layer模块

height,width = 3,3
images = np.random.uniform(size=(5,height,width,3))
output = tf.contrib.layers.avg_pool2d(images,[3,3])

output = tf.contrib.layers.convolution2d(images, num_outputs = 32,kernel_size=[3,3])

weights = tf.contrib.framework.get_variables_by_name('weights')[0]
weights_shape = weights.get_shape().as_list()

#将卷积层layers.convolution2d()和批标准化层layers.batch_norm(）结合使用
images = tf.random_uniform((5,height,width,32),seed=1)

with tf.contrib.framework.arg_scope(
    [tf.contrib.layers.convolution2d],
    normalizer_fn = tf.contrib.layers.batch_norm,
    normalizer_params = {'decay':0.9}):
    net = tf.contrib.layers.convolution2d(images,32,[3,3])
    net = tf.contrib.layers.convolution2d(net,32,[3,3])

#全连接的神经网络层fully_connected()的例子
heights, width = 3,3
inputs = tf.random_uniforms((5,height*width*3),seed=1)
with tf.name_scope('fe'):
    fc = tf.contrib.layers.fully_connected(inputs,7,
                                           outputs_collections = 'outputs',
                                           scope = 'fc')
output_collected = tf.get_collection('outputs')[0]
self.assertEquals(outputs_collected.alias,'fe/fc')

x = conv2d(x,64,[3,3],scope='conv1/conv1_1')
x = conv2d(x,64,[3,3],scope='conv1/conv1_2')
y = conv2d(x,64,[3,3],scope='conv1/conv1_3')



y = stack(x,fully_connected,[32,64,128],scope='fc')
#上面代码等同于：
x = fully_connected(x,32,scope='fc/fc_1')
x = fully_connected(x,64,scope='fc/fc_2')
y = fully_connected(x,128,scope='fc/fc_3')


####################
#损失函数

predictions = tf.constant([4,8,12,8,1,3], shape=(2,3))
targets = tf.constant([1,9,2,-5,-2,6], shape=(2,3))
weight = tf.constant([1.2,0.0],shape = [2,])

loss = tf.contrib.losses.absolute_difference(predictions,targets,weights)
#计算预测的损失值

####softmax交叉熵
predictions = tf.constant([[10.0,0.0,0.0],[0.0,20.0,0.0],[0.0,0.0,10.0]])
labels = tf.constant([[1,0,0],[0,1,0],[0,0,1]])

loss = tf.contrib.losses.softmax_cross_entropy(predictions, labels)
loss.eval()
loss.op.name

####
logits = tf.constant([[100.0,-100.0,-100.0]])
labels = tf.constant([[1,0,0]])
label_smoothing=0.1
loss = tf.contrib,losses.softmax_cross_entropy(logits,labels,
                                               label_smoothing=label_smoothing)


#####
#许多应用大部分标识的分布都比较稀疏，可以使用sparse_softmax_cross_entropy()这样计算起来会更有效率
logits = tf.constant([[10.0,0.0,0.0],0.0,10.0,0.0],[0.0,0.0,10.0])
labels = tf.constant([[0],[1],[2]],dtype = tf.int64)
loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits,labels)

#########################
#特征列Feature Column

#用类似以下的learn.datasets的API来读入数据
training_set = learn.datasets.base.load_csv(filename=iris_training,target_dtype = np.int)
test_set = learn.datasets.base.load_csv(filename=iris_testing,target_dtype = np.int)

#用layers.FeatureColumn的API定义一些特征列，例如使用real_valued_column()定义连续的特征
from tf.contrib import layers
age = layers.real_valued_column("age")
income = layers.real_valued_column("income")
spending = layers.real_valued_column("spending")
hours_of_work = layers.real_valued_column("hours_of_work")

#用sparse_column_with_keys()处理性别这样的类别特征
gender = layers.sparse_column_with_keys(column_name="gender",
                                        keys = ["female", "male]")

education = layers.sparse_column_with_hash_bucket("education",hash_bucket_size = 1000)

age_range = layers.bucketized_column(age,boundaries=[18,25,30,35,40,45,50,55,60,65])
#使用bucketized_column()将之前的年龄SparseColumn进一步的区间化，将年龄段分为18`25,26~30....

#对年龄、职业、种族这三个特征，使用cross_column建立交叉特征列
combined = layers.cross_column([age_range, race, occupation],
                               hash_bucket_size=int(1e7))

# 建立各种特征列之后，我们可以直接把他们传入不同的TF.Learn Estimator
classifier = tf.contrib.learn.LinearClassifier(feature_column=[
    gender,education,occupation,combined,age_range,race,income,
    spending], model_dir=model_dir)


###################################
#Embeddings
embedding_columns= [
    tf.contrib.layers.embedding_column(title, dimension=8),
    tf.contrib.layers.embedding_column(education, dimension=8),
    tf.contrib.layers.embedding_column(gender,dimension=8),
    tf.contrib.layers.embedding_column(race,dimension=8),
    tf.contrib.layers.embedding_column(country,dimension=8)]

est = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir = model_dir,
    linear_feature_columns = wide_columns,
    dnn_feature_columns = embadding_columns,
    dnn_hidden_units = [100,50])


indices = [[0,0],[0,1],[0,2],[1,0],[3,0],[4,0],[4,1]]
ids = [0,1,-1,-1,2,0,1]
weights = [1.0,2.0,1.0,1.0,3.0,0.0,-0.5]
shape = [5,4]

sparse_ids = tf.SparseTensor(
    tf.constant(indices,tf.int64), tf.constant(ids,tf.int64),
    tf.constant(shape,tf.int64))

sparse_weights = tf.SparseTensor(
    tf.constant(indices,tf.int64),tf.constant(weights,tf.float32),
    tf.constant(shape,tf.int64))

vocab_size = 4
embed_dim=4
num_shards = 1
embedding_weights = tf.create_partitioned_variables(
    shape=[vocab_size,embed_dim],
    slicing = [num_shards,1],
    initializer = tf.truncated_normal_initializer(mean=0.0,
                                                  stddev = 1.0/math.sqrt(vocab_size),dtype = tf.float32))
for w in embedding_weights:
    w.initializer.run()
embedding_weights = [w.eval() for w in embedding_weights]

embedding_lookup_result = (tf.contrib.layers.safe_embedding_lookup_sparse(
    embedding_weights,sparse_ids,sparse_weights).eval())



#############性能分析器tfprof
#通过以下命令安装tfprof命令行的工具
bazel build -c opt tensorflow/contrib/tfprof/...
#通过以下命令查询帮助文件
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof help

#执行互动模式，指定graph_path来分析模型的shape和参数

bazel-bin/tensorflow/contrib/tfprob/tools/tfprof/tfprof \
    --graph_path=/graph.phtxt


bazel-bin/tensorflow/contrib/tfprob/tools/tfprof/tfprof \
    --graph_path=/graph.phtxt  \
    --checkpoint_path = model.ckpt


bazel-bin/tensorflow/contrib/tfprob/tools/tfprof/tfprof \
    --graph_path=/graph.phtxt  \
    --run_meta_path=run_meta  \
    --checkpoint_path = model.ckpt

run_optins = config_pb2.RunOptions(
    trace_level = config_pb2.RunOptions.FULL_TRACE)

run_metadata = config_pb2.RunMetadata()
_ = self.sess.run(..., options=run_options, run_metadata=run_metadata)
with gfile.Open(os.path.join(output_dir,"run_meta"),"w") as f:
    f.write(run_metadata.SerializeToString())


tf.contrib.tfprof.tfprof_logger.write_op_log(graph,log_dir,op_log=None)


