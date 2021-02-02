import keras
import keras.layers
import keras.utils
from keras.models import Model

from innvestigate.utils.tests.networks import base as network_base

def build_two_heads_net(input_shape_eeg=(None,1, 640, 32), input_shape_fl=(None, 1, 250, 81*2), output_n_eeg=8, output_n_fl=32, output_n=3):
    net = {}
    net["eeg_in"] = network_base.input_layer(shape=input_shape_eeg)
    net["conv1_eeg"] = keras.layers.Conv2D(filters=32, kernel_size=(1,4), strides=(1, 2))(net["eeg_in"])
    net["pool_eeg"] = keras.layers.MaxPooling2D((1, 2))(net["conv1_eeg"])
    net["out_eeg"] = network_base.dense_layer(keras.layers.Flatten()(net["pool_eeg"]), units=output_n_eeg, activation='relu')
    net["fl_in"]=network_base.input_layer(shape=input_shape_fl)
    net["conv1_fl"] = keras.layers.Conv2D(filters=32, kernel_size=(1,4), strides=(1, 2),)(net["fl_in"])
    net["conv2_fl"] = keras.layers.Conv2D(filters=32, kernel_size=(1,2), padding='valid')(net["conv1_fl"])
    net["pool_fl"] = keras.layers.MaxPooling2D((1, 2))(net["conv2_fl"])
    net["out_fl"] = network_base.dense_layer(keras.layers.Flatten()(net["pool_fl"]), units=output_n_fl, activation='relu')
    net['concat_feat'] = keras.layers.merge.concatenate([net["out_eeg"], net["out_fl"]])
    net["concat_out"] = network_base.dense_layer(net["concat_feat"], units=output_n, activation='relu')
    net["sm_out"] = network_base.softmax(net["concat_out"])

    model_with_softmax = Model(inputs=[net['eeg_in'], net['fl_in']], outputs=net['sm_out'])
    model_without_softmax = Model(inputs=[net['eeg_in'], net['fl_in']], outputs=net['concat_out'])

    
    return model_with_softmax, model_without_softmax

def build_single_head_net(input_shape=(None, 1, 640, 81*2+32), feature_dim=32, output_n = 3):
    net = {}
    net["in"] = network_base.input_layer(shape=input_shape)
    net["conv1"] = keras.layers.Conv2D(filters=32, kernel_size=(1,4), strides=(1, 2))(net["in"])
    net["conv2"] = keras.layers.Conv2D(filters=32, kernel_size=(1,2), padding='valid')(net["conv1"])
    net["pool"] = keras.layers.MaxPooling2D((1, 2))(net["conv2"])
    net["out_feat"] = network_base.dense_layer(keras.layers.Flatten()(net["pool"]), units=feature_dim, activation='relu')
    net["out"]= network_base.dense_layer(net["out_feat"], units=output_n, activation='relu')
    net["sm_out"] = network_base.softmax(net["out"])

    model_with_softmax = Model(inputs=net['in'], outputs=net['sm_out'])
    model_without_softmax = Model(inputs=net['in'], outputs=net['out'])

  return model_with_softmax, model_without_softmax