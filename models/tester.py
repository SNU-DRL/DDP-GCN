# @Time     : Jan. 10, 2019 17:52
# @Author   : Veritas YIN
# @FileName : tester.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time

import json


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError('ERROR: the value of n_pred {} exceeds the length limit.'.format(n_pred))

    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)

    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    # update the metric on test set, if model's performance got improved on the validation.
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        min_val = evl_pred
    return min_va_val, min_val

import tensorflow.contrib.slim as slim
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def model_test(inputs, batch_size, n_his, n_pred, output_filepath, inf_mode='merge'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    load_path = pjoin(output_filepath, 'models')
    print(load_path)

    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin('{}.meta'.format(model_path)))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print('>> Loading saved model from {} ...'.format(model_path))
        # print(model_summary())

        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            # step_idx = tmp_idx = np.arange(1, n_pred + 1, 1) - 1
        else:
            raise ValueError('ERROR: test mode "{}" is not defined.'.format(inf_mode))

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl = evaluation(x_test[:, step_idx + n_his, :, :], y_test, x_stats)
        print(evl)

        np.save(pjoin(output_filepath, 'models', 'truth.npy'), x_test[:, step_idx + n_his, :, :])
        np.save(pjoin(output_filepath, 'models', 'predictions.npy'), y_test)


        # result = {evl}
        result = {}
        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            result.update({str(ix + 1): {'mape': te[0], 'mae': te[1], 'rmse': te[2]}})

            print('Time Step %i: MAPE %.3f; MAE  %.3f; RMSE %.3f.' % (ix + 1, te[0], te[1], te[2]))
        print('Model Test Time %.3f s' % (time.time() - start_time))

        with open(pjoin(output_filepath, 'performance_test.json'), 'w') as f:
            json.dump(result, f)

    print('Testing model finished!')

    return y_test


# def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
#     '''
#     Multi_prediction function.
#     :param sess: tf.Session().
#     :param y_pred: placeholder.
#     :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
#     :param batch_size: int, the size of batch.
#     :param n_his: int, size of historical records for training.
#     :param n_pred: int, the length of prediction.
#     :param step_idx: int or list, index for prediction slice.
#     :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
#     :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
#             len_ : int, the length of prediction.
#     '''
#     pred_list = []
#     for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
#         # Note: use np.copy() to avoid the modification of source data.
#         test_seq = np.copy(i[:, 0:n_his + 1, :, :])
#         step_list = []
#         for j in range(n_pred):
#             pred = sess.run(y_pred,
#                             feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
#             if isinstance(pred, list):
#                 pred = np.array(pred[0])
#             # test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
#             # test_seq[:, n_his - 1, :, :] = pred
#             test_seq[:, :n_his, :, :] = test_seq[:, 1:(n_his+1), :, :]
#             test_seq[:, n_his, :, :] = pred
#             step_list.append(pred)
#         pred_list.append(step_list)
#     #  pred_array -> [n_pred, batch_size, n_route, C_0)
#     pred_array = np.concatenate(pred_list, axis=1)
#     return pred_array[step_idx], pred_array.shape[1]
#
#
# def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx):
#     '''
#     Model inference function.
#     :param sess: tf.Session().
#     :param pred: placeholder.
#     :param inputs: instance of class Dataset, data source for inference.
#     :param batch_size: int, the size of batch.
#     :param n_his: int, the length of historical records for training.
#     :param n_pred: int, the length of prediction.
#     :param step_idx: int or list, index for prediction slice.
#     :param min_va_val: np.ndarray, metric values on validation set.
#     :param min_val: np.ndarray, metric values on test set.
#     '''
#     x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
#
#     if n_his + n_pred > x_val.shape[1]:
#         raise ValueError('ERROR: the value of n_pred {} exceeds the length limit.'.format(n_pred))
#
#     y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
#     evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)
#
#     # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
#     # chks = evl_val < min_va_val
#     # update the metric on test set, if model's performance got improved on the validation.
#     # if sum(chks):
#     #     min_va_val[chks] = evl_val[chks]
#     y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
#     evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
#         # min_val = evl_pred
#     return evl_val, evl_pred
#
#
# def model_test(inputs, batch_size, n_his, n_pred, inf_mode, output_filepath):
#     '''
#     Load and test saved model from the checkpoint.
#     :param inputs: instance of class Dataset, data source for test.
#     :param batch_size: int, the size of batch.
#     :param n_his: int, the length of historical records for training.
#     :param n_pred: int, the length of prediction.
#     :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
#     :param load_path: str, the path of loaded model.
#     '''
#     load_path = pjoin(output_filepath, 'models')
#     print(load_path)
#
#     start_time = time.time()
#     model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
#
#     test_graph = tf.Graph()
#
#     with test_graph.as_default():
#         saver = tf.train.import_meta_graph(pjoin('{}.meta'.format(model_path)))
#
#     with tf.Session(graph=test_graph) as test_sess:
#         saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
#         print('>> Loading saved model from {} ...'.format(model_path))
#
#         pred = test_graph.get_collection('y_pred')
#
#         if inf_mode == 'sep':
#             # for inference mode 'sep', the type of step index is int.
#             step_idx = n_pred - 1
#             tmp_idx = [step_idx]
#         elif inf_mode == 'merge':
#             # for inference mode 'merge', the type of step index is np.ndarray.
#             step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
#         else:
#             raise ValueError('ERROR: test mode "{}" is not defined.'.format(inf_mode))
#
#         x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
#
#         y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
#         evl = evaluation(x_test[:, step_idx + n_his, :, :], y_test, x_stats)
#
#         result = {}
#         for ix in tmp_idx:
#             te = evl[ix - 2:ix + 1]
#             result.update({str(ix + 1): {'mape': te[0], 'mae': te[1], 'rmse': te[2]}})
#
#             print('Time Step %i: MAPE %.3f; MAE  %.3f; RMSE %.3f.' % (ix + 1, te[0], te[1], te[2]))
#         print('Model Test Time %.3f s' % (time.time() - start_time))
#
#         with open(pjoin(output_filepath, 'performance_test.json'), 'w') as f:
#             json.dump(result, f)
#
#     print('Testing model finished!')
#
#     return y_test
#
#
# # def model_test(inputs, batch_size, n_his, n_pred, inf_mode, output_filepath):
# #
# #     load_path = pjoin(output_filepath, 'models')
# #     print(load_path)
# #
# #     start_time = time.time()
# #     model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
# #
# #     test_graph = tf.Graph()
# #
# #     with test_graph.as_default():
# #         saver = tf.train.import_meta_graph(pjoin('{}.meta'.format(model_path)))
# #
# #     with tf.Session(graph=test_graph) as test_sess:
# #         saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
# #         print('>> Loading saved model from {} ...'.format(model_path))
# #
# #         pred = test_graph.get_collection('y_pred')
# #
# #         x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
# #         y_test = test_sess.run(pred, feed_dict={'data_input:0': x_test, 'keep_prob:0': 1.0})
# #
# #         result = evaluation(x_test[:, n_his:, :, :], np.array(y_test)[0], x_stats)
# #
# #         # result = {}
# #         # for ix in tmp_idx:
# #         #     te = evl[ix - 2:ix + 1]
# #         #     result.update({str(ix + 1): {'mape': te[0], 'mae': te[1], 'rmse': te[2]}})
# #         #
# #         #     print('Time Step %i: MAPE %.3f; MAE  %.3f; RMSE %.3f.' % (ix + 1, te[0], te[1], te[2]))
# #         print('Model Test Time %.3f s' % (time.time() - start_time))
# #
# #         with open(pjoin(output_filepath, 'performance_test.json'), 'w') as f:
# #             json.dump(result, f)
# #
# #     print('Testing model finished!')
# #
# #     return y_test