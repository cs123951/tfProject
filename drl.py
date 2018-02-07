from gridworld import gameEnv
import tensorflow as tf
import numpy as np
import random
import os
from scipy import misc
# hero蓝色， goal绿色， fire 红色
# experience (s,a,r,sn)

class Qnetwork():
    def __init__(self, hidden_size, num_action):
        self.input = tf.placeholder(shape=[None, 84, 84, 3], dtype=tf.float32)
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.input, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3, num_outputs=512, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)

        self.advalue_conv, self.value_conv = tf.split(self.conv4, 2, 3)
        self.value_f = tf.contrib.layers.flatten(self.value_conv)
        self.advalue_f = tf.contrib.layers.flatten(self.advalue_conv)
        self.value_w = tf.Variable(tf.random_normal([hidden_size//2, 1]))
        self.advalue_w = tf.Variable(tf.random_normal([hidden_size//2, num_action]))
        self.value = tf.matmul(self.value_f,self.value_w)
        self.advantage = tf.matmul(self.advalue_f, self.advalue_w)

        self.qout = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
        self.predict = tf.argmax(self.qout, 1)

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_action, dtype=tf.float32)
        self.q = tf.reduce_sum(tf.multiply(self.qout, self.actions_onehot),
                               reduction_indices=1)

        self.targetq = tf.placeholder(shape=[None], dtype=tf.float32)
        self.sq_error = tf.square(self.targetq - self.q)
        self.loss = tf.reduce_mean(self.sq_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updated = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, size):
        # return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
        rnd = np.random.randint(0,len(self.buffer),size)
        sample_experience = np.array(self.buffer)[rnd]
        # state, action, reward, next_state, done
        sample_state = np.zeros([size, 84,84,3])
        sample_action = np.ones([size])
        sample_reward = np.zeros([size])
        # sample_done = np.ones([size])
        sample_next_state = np.zeros([size, 84,84,3])
        for i in range(size):
            sample_state[i,...] = sample_experience[i,0]
            sample_action[i] = sample_experience[i,1]
            sample_reward[i] = sample_experience[i,2]
            sample_next_state[i,...] = sample_experience[i,3]
            # sample_done[i] = sample_experience[i,4]
        return sample_state, sample_action, sample_reward, sample_next_state
        # return np.reshape(sample_experience, [size,5])
        # sample_experience = random.sample(self.buffer, size)
        # print(np.array(self.buffer).shape,len(self.buffer[0]))
        # # print(sample_experience.shape)
        # return np.reshape(np.array(sample_experience), [size, 5])

def updateTargetQ(tfvars, rate=0.001):
    total_vars = len(tfvars)
    op_holder = []
    for idx, var in enumerate(tfvars[0:total_vars//2]):
        op_holder.append(tfvars[idx+total_vars//2].assign(
            (var.value()*rate)+((1-rate)*tfvars[idx+total_vars//2].value())
                            ))
    return op_holder


hidden_size = 512
num_episode = 10000
env = gameEnv(size=5)
main_q = Qnetwork(hidden_size=hidden_size, num_action=env.actions)
target_q = Qnetwork(hidden_size=hidden_size, num_action=env.actions)
init = tf.global_variables_initializer()

num_action = env.actions
max_steps = 50
pre_episode = 200
batch_size = 32
discount = 0.99
update_step = 4
start_epsilon = 1
end_epsilon = 0.1
epsilon_step = (start_epsilon - end_epsilon)/10000
print(epsilon_step)
epsilon = start_epsilon
trainable = tf.trainable_variables()
buffer = experience_buffer(buffer_size=50000)
op_holder = updateTargetQ(trainable)
ckpt_file = "drl/drl.ckpt"
with tf.Session() as sess:
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    # if os._exists(ckpt_file):
    #     saver.restore(sess, ckpt_file)
    sess.run(init)
    for op in op_holder:
        sess.run(op)
    total_rewards = 0
    loss_value = 0
    value_value = 0
    advalue_value = 0
    for i in range(num_episode):
        # episode_buffer = experience_buffer()
        state = env.reset()
        # state = np.reshape(state, [84*84*3])
        j = 0
        loss_value = 0
        while j < max_steps:
            j += 1
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, num_action)
            else:
                action = sess.run(main_q.predict, feed_dict={main_q.input: np.reshape(state, [-1,84,84,3])})[0]
            next_state, reward, done = env.step(action)


            # next_state = np.reshape(next_state, [84*84*3])
            buffer.add([state, action, reward, next_state])

            if i > pre_episode:
                if epsilon > end_epsilon:
                    epsilon -= epsilon_step

                if (i*max_steps+j) % update_step == 0:
                    sample_state, sample_action, sample_reward, sample_next_state = buffer.sample(batch_size)
                    # for ii in range(32):
                    #     misc.imsave('pic/' + str(ii) + 'state.jpg', sample_state[ii, ...])
                    #     misc.imsave('pic/'+str(ii)+'state_next.jpg',sample_next_state[ii,...])

                    next_action, next_main_qout = sess.run([main_q.predict, main_q.qout],feed_dict={main_q.input: sample_next_state})
                    next_target_action, next_qout = sess.run([target_q.predict, target_q.qout],feed_dict={target_q.input: sample_next_state})

                    double_q = next_qout[range(batch_size), next_action]
                    double_q = np.reshape(double_q, [len(double_q)])
                    target_qvalue = sample_reward + discount*double_q

                    _, loss_value, q, conv1, conv2, conv3, conv4, value_w, advalue_w = sess.run(
                        [main_q.updated, main_q.loss, main_q.q, main_q.conv1, main_q.conv2, main_q.conv3, main_q.conv4, main_q.value_w, main_q.advalue_w],
                                            feed_dict={main_q.input: sample_state,
                                                        main_q.targetq: target_qvalue,
                                                        main_q.actions: sample_action})

                    for op in op_holder:
                        sess.run(op)
                    # print(loss_value)
            total_rewards += reward
            state = next_state
            if done == True:
                break

        # print(total_rewards)
        # total_rewards = 0
        # buffer.add(episode_buffer.buffer)
        if i %25== 0:
            print("episode %d, rewards:%.4f, epsilon:%.4f, loss:%f "%(i,total_rewards/25, epsilon, loss_value))

            # print(value_value)
            # print(advalue_value)
            total_rewards = 0
        if i % 1000==0:
            print("saving ckpt.....")
            saver.save(sess,ckpt_file)

