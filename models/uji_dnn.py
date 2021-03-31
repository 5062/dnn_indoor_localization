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


def build_dnn(input_tensor, building_units=3, floor_units=5, coord_units=2):
    x = input_tensor
    sdae_model = sdae(
        input_data=rss_train,
        hidden_layers=config['sdae_hidden_layers'],
        cache=True,
        model_fname=None,
        optimizer=config['optimizer'],
        corruption_level=config['corruption_level'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split'],
    )
    x = sdae_model(x)

    # common hidden layers
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(config['dropout'])(x)
    for units in config['common_hidden_layers']:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(config['dropout'])(x)
    common_hl_output = x  # (,64)

    # building classification output
    for units in config['building_hidden_layers']:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(config['dropout'])(x)
    x = layers.Dense(building_units)(x)
    x = layers.BatchNormalization()(x)
    building_output = layers.Activation('softmax', name='building')(x)  # (,3)

    # floor classification output
    x = layers.concatenate([common_hl_output, building_output])
    for units in config['floor_hidden_layers']:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(config['dropout'])(x)
    x = layers.Dense(floor_units)(x)
    x = layers.BatchNormalization()(x)
    floor_output = layers.Activation('softmax', name='floor')(x)  # (,5)

    # coordinates regression output
    x = layers.concatenate([common_hl_output, building_output, floor_output])
    for units in config['coordinates_hidden_layers']:
        x = layers.Dense(units, kernel_initializer='normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(config['dropout'])(x)
    x = layers.Dense(coord_units, kernel_initializer='normal')(x)
    x = layers.BatchNormalization()(x)
    coordinates_output = layers.Activation('linear', name='coord')(x)  # (,2)

    model = keras.Model(
        inputs=input_tensor,
        outputs=[building_output, floor_output, coordinates_output]
    )
    model.compile(
        optimizer=config['optimizer'],
        loss=[
            'categorical_crossentropy',
            'categorical_crossentropy',
            'mse'
        ],
        loss_weights={
            'building': config['building_weight'],
            'floor': config['floor_weight'],
            'coord': config['coordinates_weight']
        },
        metrics={
            'building': 'categorical_accuracy',
            'floor': 'categorical_accuracy',
            'coord': 'mse'
        }
    )
    return model


if __name__ == '__main__':
    config = {
        'batch_size': 256,
        'epochs': 100,
        'dropout': 0.5,
        'optimizer': "adam",
        'learning_rate': 1e-4,
        'validation_split': 0.2,
        'corruption_level': 0.1,
        'sdae_hidden_layers': [1024, 1024, 1024],
        'common_hidden_layers': [1024],
        'building_hidden_layers': [256],
        'floor_hidden_layers': [256],
        'coordinates_hidden_layers': [256],
        'building_weight': 1,
        'floor_weight': 1,
        'coordinates_weight': 1,
        'verbose': 2,
    }

    training_data, testing_data = read_data()
    rss_train = training_data.rss_scaled
    labels_train = training_data.labels
    coord_train = training_data.coord_scaled
    coord_scaler = training_data.coord_scaler
    rss_test = testing_data.rss_scaled
    labels_test = testing_data.labels
    coord_test = testing_data.coord

    input_tensor = keras.Input(shape=(rss_train.shape[1]))
    model = build_dnn(input_tensor)
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
        shuffle=True
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
