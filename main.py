# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os
from os.path import join as pjoin

import tensorflow as tf
from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import json
import argparse

n_trial = 1

gpu_id = 0
model_filename = 'model-Urban1.json'
with open(model_filename, 'r') as f:
    model_json = json.load(f)

progress = 0
for key in sorted(model_json.keys()):
    progress += 1
    print("######################## PROGRESS: {} / {} ########################".format(progress, len(model_json)))
    for trial_no in range(n_trial):
        parser = argparse.ArgumentParser()
        ## fixed ##
        parser.add_argument("--n_his", type=int, default=12)
        parser.add_argument("--n_pred", type=int, default=12)
        parser.add_argument("--batch_size", type=int, default=25)
        parser.add_argument("--epoch", type=int, default=100)
        parser.add_argument("--save", type=int, default=50)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--opt", type=str, default="RMSProp")
        parser.add_argument("--graph", type=str, default="default")
        parser.add_argument("--inf_mode", type=str, default="merge")
        parser.add_argument("--sigma", type=float, default=1e06)
        parser.add_argument("--epsilon", type=float, default=0)
        parser.add_argument("--spat_layernorm", type=bool, default=True)
        parser.add_argument("--temp_layernorm", type=bool, default=True)
        parser.add_argument("--out_layernorm", type=bool, default=True)

        parser.set_defaults(**model_json[key])

        args = parser.parse_args()
        output_filepath = os.path.join("./output", args.output_filepath, "trial_%i" %trial_no)
        print(output_filepath)
        print("Training configs: {}".format(args))

        n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
        blocks = args.blocks

        # Partition filters applied
        def addKernels(flist, kernel_name, return_itself=[]):
            distance = weight_matrix(pjoin("./dataset", "distance_W_{}.csv".format(args.data_type)),
                                     sigma=args.sigma, epsilon=args.epsilon)
            Lk = []
            for d in flist:
                if d in return_itself:
                    W = distance
                    print("%s is returned itself!" % (d))
                else:
                    mask = pd.read_csv(pjoin("./dataset", d), header=None).values
                    assert np.sum(mask == distance) < (args.n_route * args.n_route), "Distance should be returned itself!"
                    W = np.multiply(mask, distance)
                Lk.append(first_approx(W, n, symmetric=False))

            Lk = np.array(Lk).transpose((1, 2, 0)).reshape([n, n * len(flist)])
            tf.add_to_collection(name=kernel_name, value=tf.cast(tf.constant(Lk), tf.float32))
            print("{} ADDED.".format(kernel_name))

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        for k in args.Ks_dict:
            if k in ["distance_kernel", "distance"]: # return distance itself (without partition filters)
                distance = weight_matrix(pjoin("./dataset", "distance_W_{}.csv".format(args.data_type)),
                                         sigma=args.sigma, epsilon=args.epsilon)
                if len(args.Ks_dict[k]["files"]) < args.Ks_dict[k]["n"]:
                    W = scaled_laplacian(distance)
                    Lk = cheb_poly_approx(W, args.Ks_dict[k]["n"], n)
                else:
                    Lk = first_approx(distance, n, symmetric=False)
                tf.add_to_collection(name=k, value=tf.cast(tf.constant(Lk), tf.float32))
                print("{} ADDED.".format(k))

            else:
                addKernels(flist=args.Ks_dict[k]["files"], kernel_name=k,
                           return_itself=["distance_W_{}.csv".format(args.data_type),
                                          "identity_{}.csv".format(args.data_type)])

        dataset = data_gen(args.data_type, forecasting_horizon=args.n_pred, seq_len=args.n_his)
        print(">> Loading dataset with Mean: %.2f, STD: %.2f" % (dataset.mean, dataset.std))

        if not os.path.exists(os.path.join(output_filepath, "performance_test.json")):
            model_train(dataset, blocks, args, args.Ks_dict, args.ordering, output_filepath)

        if not os.path.exists(os.path.join(output_filepath, "models", "predictions.npy")):
            model_test(dataset, 50, n_his, n_pred, output_filepath)

        tf.reset_default_graph()