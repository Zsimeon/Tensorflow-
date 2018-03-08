tf.logging.set_verbosity(tf.logging.INFO)

# 怎么使用Monitor
import numpy as np
import tensorflow as tf

iris_train = tf.contrib.learn.datasets.base.load_csv(
    filename="iris_training.csv",target_dtype=np.int)
iris_test = tf.contrib.learn.datasets.base.load_csv(
    filename="iris_test.csv",target_dtype=np.int)

validation_merics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                     "precision":tf.contrib.metrics.streaming_precision,
                     "recall":tf.contrib.metrics.streaming_recall}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    iris_test.data,
    iris_test.target,
    every_n_steps=50,
    metrics = validation_merics,
    early_stopping_metric = "loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units = [10,15,10],
                                            n_classes = 3,
                                            model_dir="/iris_model_dir",
                                            config=tf.contrib.learn.RunConfig(save_checkpoints_secs = 2))

classifier.fit(x = iris_train.data,y=iris_train.target, steps=1000,
               monitors = [validation_monitor])

accuracy_score = classifier.evaluate(x=iris_test.data,
                                     y = iris_test.target)["accuracy"]

new_samples = np.array([[5.2,3.1,6.5,2.2],[2.8,3.2,5.5,3.3]],dtype = float)
y = classifier.predict(new_samples)

#在tensorboard中可视化
$tensorboard --logdir=/iris_model_dir/
Starting Tensorboard 22 on port 6006

