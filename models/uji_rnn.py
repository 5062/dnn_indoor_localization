import os
import random
from datetime import datetime
from timeit import default_timer as timer
import numpy as np
import sys

sys.path.append('../utils')
from evaluate import evaluate
from sdae import sdae
from ujiindoorloc import read_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_rnn(input_tensor):
    sdae_model = sdae(
        input_data=rss_train,  # (-1, 520)
        hidden_layers=config['sdae_hidden_layers'],
        cache=True,
        model_fname=None,
        optimizer=config['optimizer'],
        corruption_level=config['corruption_level'],
        batch_size=config['batch_size'],
        # epochs=config['epochs'],
        epochs=100,
        validation_split=config['validation_split'],
    )
    x0 = sdae_model(input_tensor)  # (-1,1024)
    # input = layers.Reshape((x.shape[1], 1))(x)  # (-1,1024,1)
    input = layers.Reshape((1, -1))(x0)  # (-1,1,1024)

    # building classification output
    x = input
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    building_output = layers.Dense(3, activation='softmax', name='building')(x)  # (-1, 3)
    building_output_a = layers.Reshape((1, -1))(building_output)

    # floor classification output
    # x = input
    x = layers.concatenate([building_output_a, input])
    x = layers.Embedding(520, 64)(x)
    x = layers.Bidirectional(
        layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, activation='tanh'))(x)
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, activation='tanh'))(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.3, activation='tanh'))(x)
    # x = layers.BatchNormalization()(x)
    floor_output = layers.Dense(5, activation='softmax', name='floor')(x)  # (-1, 5)
    floor_output_a = layers.Reshape((1, -1))(floor_output)

    # coordinates regression output
    # x = input
    x = layers.concatenate([building_output_a, floor_output_a, input])
    x = layers.Bidirectional(
        layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, activation='tanh'))(x)
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, activation='tanh'))(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.3, activation='tanh'))(x)
    # x = layers.BatchNormalization()(x)
    coordinates_output = layers.Dense(2, activation='linear', name='coord')(x)  # (-1, 2)

    model = keras.Model(
        inputs=input_tensor,
        outputs=[building_output, floor_output, coordinates_output]
    )
    model.compile(
        optimizer=config['optimizer'],
        loss={
            'building': 'categorical_crossentropy',
            'floor': 'categorical_crossentropy',
            'coord': 'mse'
        },
        loss_weights=[
            config['building_weight'],
            config['floor_weight'],
            config['coordinates_weight']
        ],
        metrics={
            'building': 'categorical_accuracy',
            'floor': 'categorical_accuracy',
            'coord': 'mse'
        }
    )
    return model


if __name__ == '__main__':
    config = {
        'num_runs': 1,
        'batch_size': 128,
        'epochs': 50,
        'optimizer': 'nadam',
        'learning_rate': 3e-4,
        'validation_split': 0.2,
        'sdae_hidden_layers': [1024, 1024, 1024],
        'corruption_level': 0.1,
        'building_weight': 1,
        'floor_weight': 1,
        'coordinates_weight': 1,
        'verbose': 2,
    }

    set_random_seed()

    training_data, testing_data = read_data()
    rss_train = training_data.rss_scaled
    labels_train = training_data.labels
    coord_train = training_data.coord_scaled
    coord_scaler = training_data.coord_scaler
    rss_test = testing_data.rss_scaled
    labels_test = testing_data.labels
    coord_test = testing_data.coord

    input_tensor = keras.Input(shape=(rss_train.shape[1]))
    model = build_rnn(input_tensor)
    # model.summary()

    weights_file = './tmp/best_weights.h5'
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        weights_file, monitor='val_loss', save_best_only=True, verbose=0)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=config['verbose'])
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True, verbose=config['verbose'])
    log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    start = timer()
    history = model.fit(
        x=rss_train,
        y=[labels_train.building, labels_train.floor, coord_train],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        verbose=config['verbose'],
        callbacks=[
            # checkpoint_cb,
            # reduce_lr_cb,
            # early_stopping_cb,
            # tensorboard_cb,
        ],
        validation_split=config['validation_split'],
    )
    elapsed_time = timer() - start
    print(" completed in {0:.4e} s".format(elapsed_time))
    model.load_weights(weights_file)

    results = evaluate(model,
                       batch_size=config['batch_size'],
                       elapsed_time=elapsed_time,
                       rss=rss_test,
                       blds_true=labels_test.building,
                       flrs_true=labels_test.floor,
                       coord_true=coord_test,
                       coord_scaler=coord_scaler)
    print(results)
