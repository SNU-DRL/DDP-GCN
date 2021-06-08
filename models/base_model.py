# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(inputs, n_his,
                Ks_dict, ordering,
                Kt, blocks, keep_prob, batch_size, do_layer_norm=[True, True, True]):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks_dict, ordering, Kt, channels, i, keep_prob, do_layer_norm)
    # Output Layer
    y = output_layer(x, 'output_layer')

    # train_loss = tf.nn.l2_loss(y - inputs[:, n_his:, :, :])
    # tf.add_to_collection(name='y_pred', value=y)
    # return train_loss, y
    tf.add_to_collection(name='copy_loss',
                         value=(tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :])) / batch_size)

    train_loss = (tf.nn.l2_loss(y[:, 0, :, :] - inputs[:, n_his, :, :])) / batch_size
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)

    return train_loss, single_pred


def model_save(sess, global_steps, model_name, output_filepath):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''

    save_path = pjoin(output_filepath, 'models')
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(prefix_path)

    print('<< Saving model to {} ...'.format(prefix_path))
