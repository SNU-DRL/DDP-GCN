# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import tensorflow as tf

def gconv(x, theta, Ks, c_in, c_out, graph_kernel_name='graph_kernel'):
    print("GRAPH CONVOLUTION: {}".format(graph_kernel_name))
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection(graph_kernel_name)[0]
    n = tf.shape(kernel)[0]
    # x -> [batch_size*T, c_in, n_route] -> [batch_size*T*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*T*c_in, Ks*n_route] -> [batch_size*T, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size*T, n_route, c_in, K_s] -> [batch_size*T*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*T*n_route, c_out] -> [batch_size*T, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv

def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x

def fc(input, c_in, c_out, name="fc"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([c_in, c_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[c_out]), name="B")
    return tf.nn.relu(tf.matmul(input, W)) + b

def temporal_block(x, Kt, c_in, c_out, do_layer_norm=True):
    _, T, n, _ = x.get_shape().as_list()

    wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
    bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
    x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='SAME') + bt
    if do_layer_norm:
        return layer_norm(tf.nn.relu(x_conv), scope='layer_norm_temp')
    else:
        return tf.nn.relu(x_conv)

def spatial_block(x, Ks_dict, c_in, c_out, ordering, scope, do_layer_norm=True):
    with tf.variable_scope(scope):
        _, T, n, _ = x.get_shape().as_list()

        weights = dict()
        idx_k = 0
        if ordering == "parallel":
            for k in Ks_dict:
                weights.update(
                    {("ws_" + k): tf.get_variable(name="ws_" + k, shape=[Ks_dict[k]['n'] * c_in, c_out], dtype=tf.float32)})
        else:
            for k in ordering:
                if idx_k == 0:
                    weights.update({("ws_" + k): tf.get_variable(name="ws_" + k, shape=[Ks_dict[k]['n'] * c_in, c_out], dtype=tf.float32)})
                else:
                    weights.update({("ws_" + k): tf.get_variable(name="ws_" + k, shape=[Ks_dict[k]['n'] * c_out, c_out], dtype=tf.float32)})
                idx_k += 1

        bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32) #we assumed to add same bias term

        idx_k_ = 0
        if ordering == "parallel":
            x_gc = []
            for k in Ks_dict:
                x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), weights["ws_" + k], Ks_dict[k]['n'], c_in, c_out, graph_kernel_name=k) + bs
                x_gc.append(tf.reshape(x_gconv, [-1, T, n, c_out]))
                if do_layer_norm:
                    x_gc.append(layer_norm(x_gconv, 'layer_norm_{}'.format(k)))
            x_gc = tf.reduce_sum(x_gc, axis=0)
            x_gc = tf.reshape(x_gc, [-1, T, n, c_out])
        else:
            for k in ordering:
                if idx_k_ == 0:
                    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), weights["ws_" + k], Ks_dict[k]['n'], c_in, c_out, graph_kernel_name=k) + bs
                else:
                    x_gconv = gconv(tf.reshape(x_gconv, [-1, n, c_out]), weights["ws_" + k], Ks_dict[k]['n'], c_out, c_out, graph_kernel_name=k) + bs
                if do_layer_norm:
                    x_gconv = layer_norm(tf.reshape(x_gconv, [-1, T, n, c_out]), 'layer_norm_{}'.format(k))
                idx_k_ += 1
            x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])

        # x_g -> [batch_size, time_step, n_route, c_out]
        return tf.nn.relu(x_gc[:, :, :, 0:c_out])

def st_conv_block(x, Ks_dict, ordering,
                  Kt, channels, scope, keep_prob, do_layer_norm=[True, True, True]):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels

    with tf.variable_scope('stn_block_{}_in'.format(scope)):
        print("INPUT", x.shape)
        x_s = spatial_block(x, Ks_dict, c_si, c_t, ordering, "spatial_conv_seq", do_layer_norm[0])
        print("SPAT", x_s.shape)
        x_t = temporal_block(x_s, Kt, c_t, c_oo, do_layer_norm[1])
        print("TEMP", x_t.shape)
        if do_layer_norm[2]:
            x_ln = layer_norm(x_t, 'layer_norm_{}'.format(scope))
        else:
            x_ln = x_t
    return tf.nn.dropout(x_ln, keep_prob)


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name='w_{}'.format(scope), shape=[1, 1, channel, 1], dtype=tf.float32)
    b = tf.get_variable(name='b_{}'.format(scope), initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, scope):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()
    x_fc = fully_con_layer(x, n, channel, scope)
    return x_fc


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean_{}'.format(v_name), mean)

        with tf.name_scope('stddev_{}'.format(v_name)):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev_{}'.format(v_name), stddev)

        tf.summary.scalar('max_{}'.format(v_name), tf.reduce_max(var))
        tf.summary.scalar('min_{}'.format(v_name), tf.reduce_min(var))

        tf.summary.histogram('histogram_{}'.format(v_name), var)
