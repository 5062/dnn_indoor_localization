import itertools
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from datetime import datetime
from mean_ci import mean_ci


def evaluate(model, batch_size, elapsed_time, rss, blds_true, flrs_true, coord_true, coord_scaler):
    # calculate the classification accuracies and localization errors
    blds_pred, flrs_pred, coords_scaled_pred = model.predict(rss, batch_size=batch_size)
    bld_results = np.equal(np.argmax(blds_true, axis=1), np.argmax(blds_pred, axis=1)).astype(int)
    flr_results = np.equal(np.argmax(flrs_true, axis=1), np.argmax(flrs_pred, axis=1)).astype(int)
    bld_acc = bld_results.mean()
    flr_acc = flr_results.mean()
    coord_pred = coord_scaler.inverse_transform(coords_scaled_pred)  # inverse-scaling

    # calculate 2D localization errors
    dist_2d = np.linalg.norm(coord_true - coord_pred, axis=1)
    mean_error_2d = dist_2d.mean()
    median_error_2d = np.median(dist_2d)

    # calculate 3D localization errors
    flr_diff = np.absolute(np.argmax(flrs_true, axis=1) - np.argmax(flrs_pred, axis=1))
    z_diff_squared = (4 ** 2) * np.square(flr_diff)
    dist_3d = np.sqrt(np.sum(np.square(coord_true - coord_pred), axis=1) + z_diff_squared)
    mean_error_3d = dist_3d.mean()
    median_error_3d = np.median(dist_3d)

    LocalizationResults = namedtuple('LocalizationResults',
                                     ['bld_acc', 'flr_acc', 'mean_error_2d', 'median_error_2d', 'mean_error_3d',
                                      'median_error_3d', 'elapsed_time'])
    return LocalizationResults(bld_acc=bld_acc, flr_acc=flr_acc,
                               mean_error_2d=mean_error_2d,
                               median_error_2d=median_error_2d,
                               mean_error_3d=mean_error_3d,
                               median_error_3d=median_error_3d,
                               elapsed_time=elapsed_time)


def plot_graph(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    :param cm: (array, shape = [n, n]) a confusion matrix of integer classes
    :param class_names: (array, shape = [n]) string names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
