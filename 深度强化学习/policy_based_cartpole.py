#先安装OpenAi
pip install gym

import numpy as np
import tensorflow as tf
import gym
env = gym.make('CartPole-v0')   #创建CartPole问题的环境env

#先测试在cartpole环境中使用随机Action的表现，作为接下来对比的baseline
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0,2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
#可以看到随机策略获得的奖励总值差不多在10~40之间，均值在20~30

#策略网络简单的带有一个隐含层的MLP
H = 50
batch_size = 25
learning_rate = 1e-1
D = 4
gamma = 0.99


#定义策略网络的具体结构
observations = tf.placeholder(tf.float32, [None,D], name = "input_x")
W1 = tf.get_variable("W1", shape=[D,H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

adam = tf.trian.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

#定义函数disocunt_rewards,用来估算每一个Action对应的潜在价值discount_r
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#定义人工设置的虚拟label的placeholder--input_y
#以及每个Action的潜在价值的placeholder--advantages
input_y = tf.placeholder(tf.float32,[None,1], name = "input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")
loglik = tf.log(input_y*(input_y - probability) + \
                (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)

tvars = tf.trainable_variables()
newGrads = tf.gradients(loss,tvars)

xs,ys,drs = [],[],[]
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    #进入实验的循环，当某个batch的平均reward达到100以上时，即Agent表现良好时，调用env.render()对实验环境进行展示
    while episode_number <= total_episodes:
        if reward_sum/batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation,[1,D])

        tfprob = sess.run(probability,feed_dict={observations:x})
        action = a if np.random.uniform() < tfprob else 0

        xs.append(x)
        y = 1-action
        ys.append(y)

        observation,reward,done,info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            eposide_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs,ys,drs = [],[],[]
            #epx、epy、epr即为一次实验中获得的所有的observation、label、reward的列表

            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            tGrad = sess.run(newGrads, feed_dict={observations:epx,
                                                  input_y:epy, advantages:discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],
                                                W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print('Average reward for episode %d : %f.' % \
                      (episode_number,reward_sum/batch_size))

                if reward_sum/batch_size > 200:
                    print("Task solved in",episode_number,'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()
            



