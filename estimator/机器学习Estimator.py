
#线性/逻辑回归
#在input_fn理创建两个特征列的数据，分别是年龄和语言，以及他们的标识
def input_fn():
    return{
        'age':tf.constant([1]),
        'language':tf.SpareTensor(values=['english'],
                                  indices=[[0,0]],
                                  shape=[1,1])
    }, tf.constant([[1]])

language = tf.contrib.layers.sparse_column_with_hash_bucket('language',100)
age = tf.contrib.layers.real_valued_column('age')

#将这些特征列传入LinearClassifier建立逻辑回归分类器
classifier = tf.contrib.learn.LinearClassifier(
    feature_columns = [age,language])
classifier.fit(input_fn = input_fn, steps=100)
classifier.evaluate(input_fn=input_fn,steps=1)['loss']
classifier.get_variable_names()

#也可以使用自定义的优化函数
classifier = tf.contrib.learn.LinearClassifier(
    n_classes = 3,
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1),
    feature_columns = [feature_column]
)

###################################################
#随机森林

#用iris数据及随机森林Estimator进行分类
hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees = 3,max_nodes = 1000, num_classes = 3, num_features = 4)
classifier = tf.contrib.learn.TensorForestEstimator(hparams)

iris = tf.contrib.learn.datasets.load_iris()
data = iris.data.astype(np.float32)
target = iris.target.astype(np.float32)
classifier.fit(x = data, y = terget, steps = 100)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

#防止随机森林过拟合的方法是损失减少的速度变慢或完全停止减少的情况下，提前停止模型的训练
from tensorflow.contrib.learn.python.learn.estimators import random_forest

early_stopping_rounds = 100check_every_n_steps = 100
monitor = random_forest.LossMonitor(early_stopping_rounds,
                                    check_every_n_steps)
classifier.fit(x = mnist.train.images,y=mnist.train.labels,batch_size = 1000,monitors=[monitor])
results = estimator.evaluate(x=mnist.test.images,y=mnist.test.labels,batch_size=1000)

########################################
#K均值聚类
import numpy as np

def make_random_centers(num_centers,num_dims):
    return np.round(np.random.rand(num_centers,
                                   num_dims).astype(np.float32)*500)

def make_random_points(centers, num_points, max_offset=20):
    num_centers,num_dims = centers.shape
    assignments = np.random.choice(num_centers,num_points)
    offsets = np.round(np.random.randn(num_points,
                                       num_dims).astype(np.float32)*max_offset)
    return (centers[assignments] + offsets,
            ssignments,
            np.add.reduce(offsets * offsets,1))

#以上两个函数时利用Numpy制造比较适合做聚类的一组数据。
#我们生成二维的10000个点，6个随机的聚类中心点
num_centers = 6
num_dims = 2
num_points = 10000
true_centers = make_random_centers(num_centers,num_dims)
points, _, scores = make_random_points(true_centers,num_points)


from tensorflow.contrib.factorization.python.ops import kmeans as kmeans_ops
from tensorflow.contrib.factorization.python.ops.kmeans import \
    KMeansClustering as KMeans
kmeans = KMeans(num_centers=num_centers,
                initial_clusters = kmeans_ops.RANDOM_INIT,
                use_mini_batch=False,
                config=RunConfig(tf_random_seed=14),
                random_seed=12)
kmeans.fit(x=points,steps=10,batch_size=8)
clusters = kmeans.clusters()

kmeans.predict(points,batch_size=128)
kmeans.score(points,batch_size=128)
kmeans.transform(points,batch_size=128)


####################################
#支持向量机
 def input_fn():
     return{
         'example_id':tf.constant(['1','2','3']),
         'feature1':tf.constant([[0.0],[1.0],[3.0]]),
         'feature2':tf.constant([[1.0],[-1.2],[1.0]]),
     },tf.constant([1],[0],[1])

 feature1 = tf.contrib.layers.real_valued_column('feature1')
 feature2 = tf.contrib.layers.real_valued_column('feature2')

 svm_classifier = tf.contrib.learn.SVM(feature_columns=[feature1,feature2],
                                       example_id_column='example_id',
                                       l1_regularization=0.0,
                                       l2_regularization=0.0)

 svm_classifier.fit(input_fn=input_fn,steps=30)
 metrics = svm_classifier.evaluate(input_fn=input_fn,steps=1)
 loss = metrics['loss']
 accuracy = metrics['accuracy']


 ###################################
 #DataFrame
 import tensorflow.contrib.learn.python.learn.dataframe.tensorflow_dataframe as df
 x = np.eye(20)
 tensorflow_df = df.TensorflowDataFrame.from_numpy(x, batch_size=10)

 pandas_df = pd.read_csv(data_path)
 tensorflow_df = df.TensorflowDataFrame.from_csv([data_path],enqueue_size=20,
                                                 batch_size=10,shuffle = False,
                                                 default_values=default_values)

 tensorflow_df.run(num_batched=10,graph=graph,session=sess)

 tensorflow_df.batch(batch_size=10,shuffle = true,num_threads=3)

 

