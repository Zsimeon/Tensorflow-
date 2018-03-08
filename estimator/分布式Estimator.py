#Estimator接受自定义模型，
#此处定义函数签名  （features，targets） -> （predictions, loss, train_op）



import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

def my_model(features, target):
    #features是数据的特征，targets是数据特征每一行的目标或者分类的标识。
    target = tf.onr_hot(target, 3,1,0)      #读热编码，使损失函数的计算更方便
    #使用layers.stack叠加多层layers.fully_connected完全连接的深度神经网络
    #每一层分别有10、20、10个隐藏节点
    features = layers.stack(features, layers.fully_connected,[10,20,10])
    prediction,loss = tf.contrib.learn.models.logistic_regression_zero_init(features,target)
    #使用contrib.layers.optimize_loss函数对损失值进行优化
    train_op = tf.contrib.layers.optimize_loss(
        los, tf.contrib.framwork.get_global_step(),optimizer='Adagrad',
        learning_rate = 0.1)
    return {'class':tf.argmax(prediction,1),'prob':prediction},loss,train_op

#把定义好的模型运用到比较常用的eiris数据进行分类
from sklearn import datasets,cross_validation
#从datasets引入数据，用cross_validation把数据分为训练和评估
iris = datasets.load_iris()
x_train,x_test,y_train,y_test = cross_validation.train_test_split(
    iris.data,iris.target,test_size = 0.2,random_state=35
)
#把定义好的my_model直接放进 learn.Estimator就可以使用fit和predict函数
classifier = learn.Estimator(model_fn=my_model)
classifier.fit(x_train,y_train,steps=700)

predictions = classifier.predict(x_test)


################################################################

#Estimator的_get_train_ops()的实现
predictions,loss,_=self._call_model_fn(features,targets,ModeKeys.EVAL)
result = {'loss':contrib.metrics.streaming_mean(loss)}
#先用自定义的模型对新的数据进行预测和计算损失值
#用ModeKeys.EVAL表明这个函数只会在评估时被用到
#用contrib.metrics模块里的streaming_mean对loss计算平均流
#也就是在之前计算过的平均值基础上加上这次迭代的损失值再计算平均值


class TensorForestEstimator(estimator.BaseEstimator):
    """AN estimator that can train and evaluate a random forest"""

#TF>Learn里的随机森林模型TensorForestEstimator把许多细节的实现放到了contrib.tensor_forest里
#只利用和暴露一些比较高阶的，需要用到的成分到TensorForestEstimator里
#下面的代码中，他所有的超参数都通过contrib.tensor_forest.ForestHParams被传到构造函数的params里
#然后在构造函数里使用params.fill()建造随机森林的TensorFlow图
#也就是tensor_forest.RandomForestGraphs
def __init__(self, params,device_assigner=None,model_dir=None,
             graph_builder_class=tensor_forest.RandomForestGraphs,
             master='',accuracy_metric=None,
             tf_random_seed=None,config=None):
    self.params = params.fill()
    self.accuracy_metric = (accuracy_metric or
                            ('r2' if self.params.regression else 'accuracy'))
    self.data_feeder = None
    self.device_assigner = (
        device_assigner or tensor_forest.RandomForestDeviceAssigner()
    )
    self.graph_builder_class = graph_builder_class
    self.training_args = {}

    super(TensorForestEstimator, self).__init__(model_dir=model_dir,config=config)

#由于很多实现太复杂而且通常需要非常有效率，很多细节都用C++实现了单独的Kernel
#_get_predict_ops（）函数首先使用tensor_forest内部C++实现的data_ops.ParseDataTensorOrDict（）函数
# 检测和转换读入的数据到可支持的数据类型，然后利用RandomForestGraphs的inference_graph函数得到预期的
def _get_predict_ops(self, features):
    graph_builder = self.graph_builder_class(
        self.params,device_assigner = self.device_assigner,training = False,
        **self.construction_args
    )
    features,spec = data_ops.ParseDataTensorOrDict(features)
    _assert_float32(features)
    return graph_builder.inference_graph(features, data_spec=spec)

#类似地，他的_get_train_ops()和_get_eval_ops()函数分别调用了RandomForestGraphs.training_loss()和
#RandomForestGraphs.inference_graph()函数


###################################
#RunConfig用来帮助用户调节程序运行时参数
#例如用num_cores选择使用的核的数量，用num_ps_replicas调节参数服务器的数量
#用gpu_memory_fraction控制使用的GPU存储的百分比
config = tf.contrib.learn.RunConfig(task=0,master="",
                                    gpu_memory_fraction = 0.8)
est = tf.contrib.learn.Estimator(model_fn=custom_model,config = config)

#以上例子是使用RunConfig参数的默认值在本地运行一个简单的模型，只使用一个任务ID和80%的GPU存储作为参数传到Estimator里
#RunConfig中的master参数是用来指定训练模型的主服务器地址的
#task是用来设置任务ID的，每一个任务ID控制一个训练模型参数服务器的replica


#########################################

flags = tf.app.flags        #定义一些可以从命令行传入的参数
flags.DEFINE_string("data_dir", "/tmp/census-data",
                    "Directory for storing the census data data")
flags.DEFINE_string("model_dir", "/tmp/cansus_wide_and_deep_model",
                    "Directory for storing the model")
flags.DEFINE_string("output_dir","", "Base output directory.")
flags.DEFINE_string("schedule","local_run",
                    "Schedule to run for this experiment.")
flags.DEFINE_string("master_grpc_url","",
                    "URL to master GRPC tensorflow server, e.g.,"
                    "grpc://127.0.0.1:2222")
flags.DEFINE_integer("num_parameter_servers",0,
                     "Number of parameter servers")
flags.DEFINE_integer("worker_index",0,"Worker index(>=0)")
flags.DEFINE_integer("train_steps",1000,"Number of training steps")
flags.DEFINE_integer("eval_steps",1,"Number of evaluation steps")

FLAGS = flags.FLAGS

#建立一个Experiment对象的函数
def create_experiment_fn(output_dir):
    config = run_config.RunConfig(master = FLAGS.maste_grpc_url,
                                  num_ps_replicas = FLAGS.num_parameter_severs,
                                  task=FLAGS.worker_index)

    estimator = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=FLAGS.model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_uints=[5],
        config = config
    )
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=data_source.input_train_fn,
        eval_input_fn = data_source.input_test_fn,
        train_steps = FLAGS.train_steps,
        eval_steps = FLAGS.eval_steps
    )

# 把create_experiment_fn()函数传入LearnRunner里进行不同类型的实验
learn_runner.run(experiment_fn=create_experiment_fn,
                 output_dir=FLAGS.output_dir,
                 schedulw=FLAGS.schedule)



