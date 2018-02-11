import numpy as np
import tensorflow as tf
import gym

def random_rewards(episodes = 10):
    env = gym.make('CartPole-v0')
    env.reset()
    rewards = 0
    i_episode = 0
    avg_rewards = []
    while i_episode < episodes:
    #     env.render()
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        rewards += reward
        if done:
            avg_rewards.append(rewards)
            rewards = 0
            i_episode += 1
            env.reset()

    print(avg_rewards)
    print(np.mean(avg_rewards))

def discount_rewards(r, gamma = 0.99):
    total_rewards = np.zeros_like(r)
    reward = 0
    for t in reversed(range(r.size)):
        reward = reward*gamma + r[t]
        total_rewards[t] = reward
    return total_rewards

def two_layer_network(lr = 0.1, hidden_layers = 50, episodes = 1600, batch_size = 32):
    env = gym.make('CartPole-v0')
    len_observe = env.observation_space.shape[0]
    observation = tf.placeholder(tf.float32, [None, len_observe], name = 'observe')
    w1 = tf.get_variable("w1", shape=[len_observe, hidden_layers], initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.matmul(observation, w1))
    w2 = tf.get_variable("w2", shape=[hidden_layers, 1], initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(layer1, w2)
    prob = tf.nn.sigmoid(score)

    tvars = tf.trainable_variables()
    advantages = tf.placeholder(tf.float32,name="reward_signal")
    labels = tf.placeholder(tf.float32, [None,1],name='labels')
    loglik = tf.log(labels*(labels - prob) + (1 - labels)*(labels + prob))
    loss = -tf.reduce_mean(loglik * advantages)
    newGrads = tf.gradients(loss,tvars)

    w1grad = tf.placeholder(tf.float32, name='grad_w1')
    w2grad = tf.placeholder(tf.float32, name='grad_w2')
    batch_grad = [w1grad, w2grad]
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    updategrad = opt.apply_gradients(zip(batch_grad, tvars))


    i_episode = 0
    avg_rewards = 0
    rewards_list = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ob = env.reset()
        gradbuffer = sess.run(tvars)
        for ix, tg in enumerate(gradbuffer):
            gradbuffer[ix] = tg*0
        # observation, action, reward
        obs, ats, rws = [], [], []
        while i_episode < episodes:
            input_obs = np.reshape(ob, [1,len_observe])
            prob_val = sess.run(prob, feed_dict={observation: input_obs})
            action = 1 if np.random.uniform() <prob_val else 0
            obs.append(input_obs)
            ats.append(1-action)
            ob, reward, done, info = env.step(action)
            avg_rewards += reward
            rws.append(reward)
            if done:
                rewards_list.append(avg_rewards)
                obsv = np.vstack(obs)
                atsv = np.vstack(ats)
                rwsv = np.vstack(rws)
                discount_rw = discount_rewards(rwsv)
                tgrad = sess.run(newGrads, feed_dict={observation: obsv, labels: atsv, advantages: discount_rw})
                for ix, tg in enumerate(tgrad):
                    gradbuffer[ix] += tg
                if i_episode % batch_size == 0:
                    sess.run(updategrad, feed_dict={w1grad: gradbuffer[0], w2grad: gradbuffer[1]})
                    for ix, tg in enumerate(gradbuffer):
                        gradbuffer[ix] = tg * 0
                i_episode += 1
                ob = env.reset()
                obs, ats, rws = [], [], []
                avg_rewards = 0
        print(rewards_list)
        print(np.mean(rewards_list))