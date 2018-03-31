from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams', 'batch_size, num_classes, '
                     'num_residual_units, use_bottleneck, '
                     'relu_leakiness, weight_decay_rate')


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode='train'):

        self.hps = hps  # hyper parameters
        self._images = images
        self.labels = labels
        self.mode = mode

        # self._build_model()
        self._build_model_mini()


    def _build_model_mini(self):
        with tf.variable_scope('init'):
            x = self._images
            """the first layer conv (3, 3*3/1, 16)"""
            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

        # resnet param
        strides = [1, 2, 2]
        # activate_before_res
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            # bottleneck: res unit module
            res_func = self._bottleneck_residual
            # num_channels
            filters = [16, 64, 128, 256]
        else:
            # standard res unit module
            res_func = self._residual
            # num_channels
            filters = [16, 16, 32, 64]

        # the first group
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), activate_before_residual=False)

        # the second group
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, input_channels=filters[1], output_channels=filters[2],
                         stride=self._stride_arr(strides[1]),
                         activate_before_residual=activate_before_residual[1])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        # the third group
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        # global pooling layer
        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        # fc + Softmax
        with tf.variable_scope('logit'):
            self.out = self._fully_connected(x, self.hps.num_classes)
            # self.out = tf.nn.softmax(logits)

    # transform step to tf.nn.conv2d needed
    def _stride_arr(self, stride):
        return [1, stride, stride, 1]

    # res unit module // 2 layers
    def _residual(self, x, input_channels, output_channels, stride, activate_before_residual=False):
        # if activation before res
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        # the first sub layer
        with tf.variable_scope('sub1'):
            # 3*3 conv
            x = self._conv('conv1', x, 3, input_channels, output_channels, stride)

        # the second sub layer
        with tf.variable_scope('sub2'):
            # BN&ReLU activate
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            # 3*3 conv step=1
            x = self._conv('conv2', x, 3, output_channels, output_channels, [1, 1, 1, 1])

        # merge res layer
        with tf.variable_scope('sub_add'):
            # num of channels changed
            if input_channels != output_channels:
                # average pooling, no padding zero
                orig_x = tf.nn.avg_pool(orig_x, ksize=stride, strides=stride, padding='VALID')
                # channel add zero
                orig_x = tf.pad(orig_x,
                                [[0, 0], [0, 0],
                                 [0, 0],
                                 [(output_channels - input_channels) // 2, (output_channels - input_channels) // 2]
                                 ])
            # merge res
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    # bottleneck res unit module
    def _bottleneck_residual(self, x, input_channels, output_channels, stride, activate_before_residual=False):
        # if activation before res
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        # the first sub layer
        with tf.variable_scope('sub1'):
            # 1x1 conv, use input step, channels
            x = self._conv('conv1', x, 1, input_channels, output_channels / 4, stride)

        # the second sub layer
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            # 3x3 conv, step=1, channels same
            x = self._conv('conv2', x, 3, output_channels / 4, output_channels / 4, [1, 1, 1, 1])

        # the third sub layer
        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, output_channels / 4, output_channels, [1, 1, 1, 1])

        # merge res layers
        with tf.variable_scope('sub_add'):
            if input_channels != output_channels:
                # 1x1 conv,
                orig_x = self._conv('project', orig_x, 1, input_channels, output_channels, stride)

            # merge res
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # Batch Normalization
    # ((x-mean)/var)*gamma+beta
    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            # the dim of input
            params_shape = [x.get_shape()[-1]]
            # offset
            beta = tf.get_variable('beta',
                                   params_shape,
                                   tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            # scale
            gamma = tf.get_variable('gamma',
                                    params_shape,
                                    tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                # mean and dev
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                # create batch: mean, dev
                moving_mean = tf.get_variable('moving_mean',
                                              params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance',
                                                  params_shape, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)
                # add batch mean and dev
                # moving_mean = moving_mean * decay + mean * (1 - decay)
                # moving_variance = moving_variance * decay + variance * (1 - decay)
                # self._extra_train_ops.append(moving_averages.assign_moving_average(
                #     moving_mean, mean, 0.9))
                # self._extra_train_ops.append(moving_averages.assign_moving_average(
                #     moving_variance, variance, 0.9))
            else:
                # get batch:mean dev
                mean = tf.get_variable('moving_mean',
                                       params_shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32),
                                       trainable=False)
                variance = tf.get_variable('moving_variance',
                                           params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32),
                                           trainable=False)
                # add to histogram
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            # BNlayer:((x-mean)/var)*gamma+beta
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    # weight decay, L2 regular loss
    def _decay(self):
        costs = []
        # ergodic all trainable variables
        for var in tf.trainable_variables():
            # only compute DW variable
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        #
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    # 2D conv
    def _conv(self, name, x, filter_size, input_channels, out_channels, strides, padding='SAME'):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_channels
            # get or new conv filer, init with random
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, input_channels, out_channels],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            # conv
            return tf.nn.conv2d(x, kernel, strides, padding=padding)

    # leaky ReLU
    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    # fc layer
    def _fully_connected(self, x, out_dim):
        # transform to 2D tensor, size:[N, -1]
        # x = tf.reshape(x, [self.hps.batch_size, -1])
        # param: w, avg random init, [-sqrt(3/dim), sqrt(3/dim)]*factor
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        # parm: b, 0 init
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        # x * w + b
        return tf.nn.xw_plus_b(x, w, b)

    # global ang pool
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
