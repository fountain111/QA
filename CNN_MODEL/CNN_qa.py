import tensorflow as tf

from utility.Model_Parameter import *
from utility.Global_definition import *




class InsQACNN(object):
    def __init__(self, vocab_size, sentence_length, batch_size, words_length_for_filters, embedding_size, kernel_maps):
        #1:question,2:answer,3:negative answer
        self.input_x_1 = tf.placeholder(tf.int32, [batch_size, sentence_length])
        self.input_x_2 = tf.placeholder(tf.int32, [batch_size, sentence_length])
        self.input_x_3 = tf.placeholder(tf.int32, [batch_size, sentence_length])
        #weights,also know as word embbeding
        weights_1 = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),)

        self.word_embedding_1 = tf.nn.embedding_lookup(weights_1, self.input_x_1)
        self.word_embedding_2 = tf.nn.embedding_lookup(weights_1, self.input_x_2)
        self.word_embedding_3 = tf.nn.embedding_lookup(weights_1, self.input_x_3)
        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []


        for i, words_length_for_filter in enumerate(words_length_for_filters):
            kernel= [words_length_for_filter, embedding_size, 1, kernel_maps]
            weights_cnn = weight_variable(kernel)
            b = bias_variable(kernel_maps)

            # question
            conv = conv_p_valid(self.word_embedding_1,weights_cnn)
            h = tf.nn.relu(tf.nn.bias_add(conv,b))
            pooled_outputs_1.append(max_pool_1x1(h, [1, words_length_for_filter - kernel_maps + 1, 1, 1]))

            #answer
            h = tf.nn.relu(tf.nn.bias_add(conv_p_valid(self.word_embedding_2,weights_cnn)),b)
            pooled_outputs_2.append(max_pool_1x1(h, [1, words_length_for_filter - kernel_maps + 1, 1, 1]))

            #negative
            h = tf.nn.relu(tf.nn.bias_add(conv_p_valid(self.word_embedding_3,weights_cnn)),b)
            pooled_outputs_3.append(max_pool_1x1(h, [1, words_length_for_filter - kernel_maps + 1, 1, 1]))


##
#L1 inlucde 2 models,  predict the postion of  two eyes centre by
#using the whole image of face and the half whole image.
kernel_sizes = [1,2,3,5]

input_channel = 1
kernel_maps_1= 500
kernel_maps_2= 64
kernel_maps_3= 128
kernel_maps_4= 256
kernel_maps_5= 512

# 0 :pool,parameter of weights,after pool

fc_para_1 = 512
fc_para_2 = 512
label_szie = 4





def build_deep_cnn(input_datas,kernels,is_training):
    weight  = weight_variable(kernels)
    bias    = bias_variable([kernels[3]])
    conv_ = tf.nn.bias_add(conv(input_datas, weight), bias)
    bn_conv = norm_layer(conv_, is_training)
    pool = max_pool_2x2(tf.nn.relu(bn_conv))
    return pool

def build_deep_fc(input_datas,shape,is_training,keep_prob):
        weight = weight_variable(shape)
        bias = bias_variable([shape[1]])
        z = tf.matmul(input_datas, weight) + bias
        bn_fc= norm_layer(z, is_training)
        fc = tf.nn.dropout(tf.nn.relu(bn_fc), keep_prob)
        return fc


filter_shape = [filter_size, embedding_size, 1, num_filters]
W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

def cnn_model():
    for i, kernel_size in enumerate(kernel_sizes):
        kernel_shape = [kernel_size, embedding_size, 1, kernel_maps_1]


def L1_model_with_bn(input_datas, keep_prob,is_training):
    #reshape if nedd
    kernels = [[3, 3, input_channel, kernel_maps_1],
               [3, 3, kernel_maps_1, kernel_maps_2],
               [3, 3, kernel_maps_2, kernel_maps_3],
               [3, 3, kernel_maps_3, kernel_maps_4],
               [3, 3, kernel_maps_4, kernel_maps_5]]
    number_kernels = len(kernels)
    input_layer = tf.reshape(input_datas,[-1,IMAGE_INPUT_HEIGH,IMAGE_INPUT_WIDTH,input_channel])
    for i in range(number_kernels):
        input_layer = build_deep_cnn(input_layer,kernels[i],is_training)

    #full connect
    fc_para_0 = IMAGE_INPUT_HEIGH //2**(number_kernels)* IMAGE_INPUT_WIDTH // 2**(number_kernels)   * kernel_maps_5
    fc_weigths = [[fc_para_0,fc_para_1],[fc_para_1,fc_para_2]]
    input_fc = tf.reshape(input_layer,[-1, fc_weigths[0][0]])
    for i in range(len(fc_weigths)):
        input_fc = build_deep_fc(input_fc,fc_weigths[i],is_training,keep_prob)

    #last
    fc_w = weight_variable([512, 4])
    fc_b = bias_variable([4])
    labels = tf.matmul(input_fc, fc_w) + fc_b

    return labels


