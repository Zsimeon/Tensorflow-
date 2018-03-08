def _input_fn(num_epochs=None):
    features = {'age':tf.train.limit_epochs(tf.constant([[.8],[.2],[.1]]),
                                            num_epochs=num_epochs),
                'language':tf.SparseTensor(values=['en','fr','zh'],
                                           indices=[[0,0],[0,1],[2,0]],
                                           shape=[3,2])}
    return features,tf.constant([[1],[0],[0]],dtype=tf.int32)
language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
    'language',hash_bucket_size=20)
feature_columns = [
    tf.contrib.layers.embedding_column(language_column,dimension=1),
    tf.contrib.layers.real_valued_column('age')
]

classifier = tf.contrib.learn.DNNClassifier(
    n_classes=2,
    feature_columns = feature_columns,
    hidden_units = [3,3],
    config=tf.contrib.learn.RunConfig(tf_random_seed=1)
)

classifier.fit(input_fn=_input_fn, steps=100)
scores = classifier.evaluate(_input_fn=input_fn,steps = 1)

def _iinput_fn_train():
    target = tf.constant([[1],[0],[0],[0]])
    features = {
        'x': tf.ones(shape=[4,1],dtype=tf.float32),
        'w':tf.constant([[100.],[3.],[2.],[2.]])
    }

    return features, target

classifier = tf.contrib.learn.DNNClassifier(
    weight_column_name='w',
    feature_columns = [tf.contrib.layers.real_valued_column('x')],
    hidden_units = [3,3],
    config = tf.contrib.learn.RunConfig(tf._random_seed=3)
    )

classifier.fit(input_fn = _input_fn_train,steps=100)

############################
#也可以传入我们自定义的metrics方程_my_metric_op()
def _iinput_fn_train():
    target = tf.constant([[1], [0], [0], [0]])
    features = {'x':tf.ones(shape-[4,1],dtype=tf.float32),}
    return features,target

def _my_metric_op(predictions, targets):
    predictions = tf.slice(predictions,[0,1],[-1,-1])
    return tf.reduce_sum(tf.mul(predictions,targets))

input = [[[1,1,1],[2,2,2]],
         [[3,3,3],[4,4,4]],
         [[5,5,5],[6,6,6]]]

classifier = tf.contrib.learn.DNNClassifier(
    feature_columns = [tf.contrib.layers.real_valued_column('x')],
    hidden_units=[3,3],
    config=tf.contrib.learn.RunConfig(tf_random_seed=1)
)

classifier.fit(input_fn=_input_fn_train,steps=100)

scores = classifier.evaluate(
    input_fn=_input_fn_train,
    steps=100,
    metrics = {
        'my_accuracy':tf.contrib.metrics.streaming_accuracy,
        ('my_prediction','classes'):tf.contrib.metrics.streaming_prediction,
        ('my_metric','probabilities'):_my_metric_op})

def optimizer_exp_decay():
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        learning_rate = 0.1,global_step=global_step,
        decay_steps = 100,decay_rate = 0.001    )
    return tf.train.AdagradOptimizer(learning_rate=learning_rate)

iris = datasets.load_iris()
x_train,x_test,y_train,y_test = train_tesst_split(iris.data,iris.target,
                                                  test_size=0.2,random_state=42)
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
    x_train)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units = [10,20,10],
                                            n_classes=3,
                                            optimizer=optimizer_exp_decay)

classifier.fit(x_train,y_train,steps=800)


#####################################
#广度深度模型
gender = tf.confib.layers.sparse_column_with_keys(
    "gender",keys = ["female","male"])

education = tf.contrib.layers.aprse_column_with_hash_bucket(
    "education",hash_bucker_size = 1000)
relationship = tf.contrib.alyers.sprase_column_with_hash_bucket(
    "relationship",hash_bucket_size=100
)
worklass = tf.contrib.layers.sparse_column_with_hash_bucket(
    "workclass",hash_bucket_size=100
)

wide_columns = [gender,education]
deep_columns = [relationship,workclass]

m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir = model_dir,
    linear_feature_columns = wide_columns,
    dnn_feature_columns = wide_columns,
    dnn_hidden_units = [100,50])




