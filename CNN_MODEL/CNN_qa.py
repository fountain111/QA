import tensorflow as tf

from Utility.Model_Parameter import *



words_filters_lens = [1,2,3,5]
ouput_features = 500



def inference(q,a,n,vocab_size,embedding_size,questionlen):
    W1 = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    q = tf.expand_dims(tf.nn.embedding_lookup(W1,q),-1)
    a = tf.expand_dims(tf.nn.embedding_lookup(W1,a),-1)
    n = tf.expand_dims(tf.nn.embedding_lookup(W1,n),-1)

    pool_q = []
    pool_a = []
    pool_n = []

    kernels = [[words_filters_lens[0],EMBEDDING_SIZE,1,ouput_features],
               [words_filters_lens[1],EMBEDDING_SIZE,1,ouput_features],
               [words_filters_lens[2],EMBEDDING_SIZE,1,ouput_features],
               [words_filters_lens[3],EMBEDDING_SIZE,1,ouput_features]]



    for i in  range(len(kernels)):
        W = weight_variable(kernels[i])
        b = tf.Variable(tf.constant(0.1, shape=[ouput_features]))

        #b = bias_variable(ouput_features)
        conv_q =  tf.nn.conv2d(q,W,strides=[1,1,1,1],padding='VALID')
        activation = tf.nn.relu(tf.nn.bias_add(conv_q,b))
        pool =  tf.nn.max_pool(activation,ksize=[1,questionlen - words_filters_lens[i]+ 1 ,1,1],strides=[1,1,1,1],padding='VALID')
        pool_q.append(pool)

        conv_a = tf.nn.conv2d(a, W, strides=[1, 1, 1, 1], padding='VALID')
        activation = tf.nn.relu(tf.nn.bias_add(conv_a, b))
        pool = tf.nn.max_pool(activation, ksize=[1, questionlen - words_filters_lens[i] +  1, 1, 1], strides=[1, 1, 1, 1],padding='VALID')
        pool_a.append(pool)

        conv_n = tf.nn.conv2d(n, W, strides=[1, 1, 1, 1], padding='VALID')
        activation = tf.nn.relu(tf.nn.bias_add(conv_n, b))
        pool = tf.nn.max_pool(activation, ksize=[1, questionlen - words_filters_lens[i] +  1, 1, 1], strides=[1, 1, 1, 1],padding='VALID')
        pool_n.append(pool)
    num_kernels_total = ouput_features * len(kernels)
    pool_q = tf.reshape(tf.concat(3, pool_q), [-1, num_kernels_total])
    pool_a = tf.reshape(tf.concat(3, pool_a), [-1, num_kernels_total])
    pool_n = tf.reshape(tf.concat(3, pool_n), [-1, num_kernels_total])
    return pool_q,pool_a,pool_n





#q:question,a:answer,ne:negative answer
def loss(q,a,ne):
    len_q = tf.sqrt(tf.reduce_sum(tf.mul(q, q), 1))  # 计算向量长度,Batch模式
    len_a = tf.sqrt(tf.reduce_sum(tf.mul(a, a), 1))
    len_ne = tf.sqrt(tf.reduce_sum(tf.mul( ne, ne), 1))
    dot_12 = tf.reduce_sum(tf.mul(q, a), 1)  # 计算向量的点乘,Batch模式
    dot_13 = tf.reduce_sum(tf.mul(q, ne), 1)

    cos_12 = tf.div(dot_12, tf.mul(len_q, len_a))  # 计算向量夹角,Batch模式
    cos_13 = tf.div(dot_13, tf.mul(len_q, len_ne))

    zero = tf.constant(0, shape=[BATCH_SIZE], dtype=tf.float32)
    margin = tf.constant(0.05, shape=[BATCH_SIZE], dtype=tf.float32)
    losses = tf.maximum(zero, tf.sub(margin, tf.sub(cos_12, cos_13)))
    loss_value = tf.reduce_sum(losses)
    return loss_value


def train_in_cnn(loss_op):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_op)
    return train_step












