! pip install keras==2.2.4
%tensorflow_version 1.x
! pip install innvestigate



import numpy as np
import torch
import sys
import glob
import pickle
import scipy.io
import os
import random as rn
import argparse

import keras
import keras.backend
import tensorflow as tf

from google.colab import drive

import innvestigate
import innvestigate.utils as iutils
from innvestigate.utils.tests.networks import base as network_base

drive.mount("/content/drive/")
sys.path.append('/content/drive/My Drive/')

from dataset.DEAP_keras import DEAP
from models.CNN_keras import build_single_head_net, build_two_heads_net

## Initialize seed for reproducibility
seed = 1
np.random.seed(seed)
rn.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=config)
keras.backend.set_session(sess)


def train(args):
    subject = args.subject
    num_head = args.num_head
    logname = args.logname
    lr  = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    dataset=DEAP(subject=subject)
    train_indices, valid_indices=dataset.train_valid_split()
    train_X_eeg = dataset[train_indices]['eeg']
    train_X_eeg = np.expand_dims(train_X_eeg, axis=1)
    train_X_fl = dataset[train_indices]['face']
    train_X_fl = np.expand_dims(train_X_fl, axis=1)
    train_X_concat = np.concatenate((train_X_eeg, train_X_fl), axis = -1)
    train_Y = dataset[train_indices]['label_arousal']
    valid_X_eeg = dataset[valid_indices]['eeg']
    valid_X_eeg = np.expand_dims(valid_X_eeg, axis=1)
    valid_X_fl = dataset[valid_indices]['face']
    valid_X_fl = np.expand_dims(valid_X_fl, axis=1)
    valid_X_concat = np.concatenate((valid_X_eeg, valid_X_fl), axis = -1)
    valid_Y = dataset[valid_indices]['label_arousal']

    if num_head ==1:
        model_with_softmax, model_without_softmax = build_single_head_net()
        opt = keras.optimizers.Adam(lr=lr)
        model_with_softmax.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        hitory = model_with_softmax.fit(train_X_concat, train_Y, epochs=epochs, batch_size=batch_size,)
        scores = model_with_softmax.evaluate(valid_X_concat, valid_Y, batch_size=batch_size)
        np.savetxt(logname+"_loss.txt", np.array(history), delimiter=",")
        with open(logname+'_loss.txt', 'a') as f:
            f.write("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

    elif num_head ==2:
        model_with_softmax, model_without_softmax = build_two_heads_net()
        opt = keras.optimizers.Adam(lr=lr)
        model_with_softmax.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        hitory = model_with_softmax.fit([train_X_eeg, train_X_fl], train_Y, epochs=epochs, batch_size=batch_size)
        scores = model_with_softmax.evaluate([valid_X_eeg, valid_X_fl], valid_Y, batch_size=batch_size)
        np.savetxt(logname+"_loss.txt", np.array(history), delimiter=",")
        with open(logname+'_loss.txt', 'a') as f:
            f.write("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

    if args.save_model:
        model_without_softmax.set_weights(model_with_softmax.get_weights())
        model_without_softmax.save(logname+"_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", type=int, help="subject number"
    )
    parser.add_argument(
        "--num-head", type=int, help="model type: single head/double head"
    )

    parser.add_argument("--logname", type=str, help="log name for saveing")
    parser.add_argument("--save-model", type=bool, help='save model')
    parser.add_argument("--lr", type=float, default = 0.0001, help='learning rate')
    parser.add_argument("--epochs", type = int, default = 50, help='training epochs')
    parser.add_argument("--batch-size", type = int, default = 32, help='batch size')
    train(arg)