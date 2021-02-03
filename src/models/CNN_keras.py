import keras
import keras.layers
import keras.utils
from keras.models import Model

from innvestigate.utils.tests.networks import base as network_base


def build_two_heads_net(
    input_shape_eeg=(None, 1, 640, 32),
    input_shape_fl=(None, 1, 250, 81 * 2),
    output_n_eeg=8,
    output_n_fl=8,
    output_n=3,
):
    net = {}
    net["eeg_in"] = network_base.input_layer(shape=input_shape_eeg)
    net["conv1_eeg"] = keras.layers.Conv2D(
        32, (1, 2), strides=(1, 1), activation="relu"
    )(net["eeg_in"])
    net["conv2_eeg"] = keras.layers.Conv2D(
        64, (1, 2), strides=(1, 1), activation="relu"
    )(net["conv1_eeg"])
    net["pool_eeg"] = keras.layers.MaxPooling2D((1, 2))(net["conv2_eeg"])
    net["out_eeg"] = network_base.dense_layer(
        keras.layers.Flatten()(net["pool_eeg"]), units=output_n_eeg, activation="relu"
    )
    net["fl_in"] = network_base.input_layer(shape=input_shape_fl)
    net["conv1_fl"] = keras.layers.Conv2D(
        32, (1, 2), strides=(1, 1), activation="relu"
    )(net["fl_in"])
    net["conv2_fl"] = keras.layers.Conv2D(
        64, (1, 2), strides=(1, 1), activation="relu"
    )(net["conv1_fl"])
    net["pool_fl"] = keras.layers.MaxPooling2D((1, 2))(net["conv2_fl"])
    net["out_fl"] = network_base.dense_layer(
        keras.layers.Flatten()(net["pool_fl"]), units=output_n_fl, activation="relu"
    )
    net["concat_feat"] = keras.layers.merge.concatenate([net["out_eeg"], net["out_fl"]])
    net["concat_out"] = network_base.dense_layer(net["concat_feat"], units=32)
    net["out"] = network_base.dense_layer(net["concat_out"], units=output_n)
    net["sm_out"] = network_base.softmax(net["out"])

    model_with_softmax = Model(
        inputs=[net["eeg_in"], net["fl_in"]], outputs=net["sm_out"]
    )
    model_without_softmax = Model(
        inputs=[net["eeg_in"], net["fl_in"]], outputs=net["out"]
    )

    return model_with_softmax, model_without_softmax


def build_single_head_net(
    input_shape=(None, 1, 640, 81 * 2 + 32), feature_dim=16, output_n=3
):
    net = {}
    net["in"] = network_base.input_layer(shape=input_shape)
    net["conv1"] = keras.layers.Conv2D(
        filters=64, kernel_size=(1, 2), strides=(1, 1), activation="relu"
    )(net["in"])
    net["conv2"] = keras.layers.Conv2D(
        filters=128, kernel_size=(1, 2), activation="relu"
    )(net["conv1"])
    net["pool"] = keras.layers.MaxPooling2D((1, 2))(net["conv2"])
    net["out_feat"] = network_base.dense_layer(
        keras.layers.Flatten()(net["pool"]), units=feature_dim, activation="relu"
    )
    net["out"] = network_base.dense_layer(net["out_feat"], units=output_n)
    net["sm_out"] = network_base.softmax(net["out"])

    model_with_softmax = Model(inputs=net["in"], outputs=net["sm_out"])
    model_without_softmax = Model(inputs=net["in"], outputs=net["out"])

    return model_with_softmax, model_without_softmax


def eeg_net(input_shape_eeg=(None, 1, 640, 32), output_n_eeg=8, output_n=3):
    net = {}
    net["in"] = network_base.input_layer(shape=input_shape_eeg)
    net["conv1_eeg"] = keras.layers.Conv2D(
        32, (1, 2), strides=(1, 1), activation="relu"
    )(net["in"])
    net["conv2_eeg"] = keras.layers.Conv2D(
        64, (1, 2), strides=(1, 1), activation="relu"
    )(net["conv1_eeg"])
    net["pool_eeg"] = keras.layers.MaxPooling2D((1, 2))(net["conv2_eeg"])
    net["out_eeg"] = network_base.dense_layer(
        keras.layers.Flatten()(net["pool_eeg"]), units=output_n_eeg, activation="relu"
    )
    net["out"] = network_base.dense_layer(net["out_eeg"], units=output_n)
    net["sm_out"] = network_base.softmax(net["out"])

    model_with_softmax = Model(inputs=net["in"], outputs=net["sm_out"])
    model_without_softmax = Model(inputs=net["in"], outputs=net["out"])

    return model_with_softmax, model_without_softmax


def fl_net(input_shape_fl=(None, 1, 640, 81 * 2), output_n_fl=8, output_n=3):
    net = {}
    net["in"] = network_base.input_layer(shape=input_shape_fl)
    net["conv1_fl"] = keras.layers.Conv2D(
        32, (1, 2), strides=(1, 1), activation="relu"
    )(net["in"])
    net["conv2_fl"] = keras.layers.Conv2D(
        64, (1, 2), strides=(1, 1), activation="relu"
    )(net["conv1_fl"])
    net["pool_fl"] = keras.layers.MaxPooling2D((1, 2))(net["conv2_fl"])
    net["out_fl"] = network_base.dense_layer(
        keras.layers.Flatten()(net["pool_fl"]), units=output_n_fl, activation="relu"
    )
    net["out"] = network_base.dense_layer(net["out_fl"], units=output_n)
    net["sm_out"] = network_base.softmax(net["out"])

    model_with_softmax = Model(inputs=net["in"], outputs=net["sm_out"])
    model_without_softmax = Model(inputs=net["in"], outputs=net["out"])

    return model_with_softmax, model_without_softmax


def pretrained(
    eeg_model,
    fl_model,
    input_shape_eeg=(None, 1, 640, 32),
    input_shape_fl=(None, 1, 640, 162),
):
    net = {}
    for layer in eeg_model.layers[:-4]:
        layer.trainable = False
    for layer in fl_model.layers[:-4]:
        layer.trainable = False
    net["eeg_features"] = eeg_model.layers[-3].output
    net["fl_features"] = fl_model.layers[-3].output
    net["concat_feature"] = keras.layers.merge.concatenate(
        [net["eeg_features"], net["fl_features"]]
    )
    net["out1"] = network_base.dense_layer(net["concat_feature"], units=32)
    net["out"] = network_base.dense_layer(net["out1"], units=3)

    net["sm_out"] = network_base.softmax(net["out"])
    model_with_softmax = Model(
        inputs=[eeg_model.input, fl_model.input], outputs=net["sm_out"]
    )

    return model_with_softmax