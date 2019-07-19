import os
import numpy as np
import re
from time import time
import h5py
import math

def versions(list_devices=False):
    """ Prints system info and version strings for finicky libraries. """
    import keras
    import tensorflow as tf
    import h5py
    import platform
    
    print("Platform:", platform.platform())
    print("h5py:\n" + h5py.version.info)
    # print("numpy:",np.version.full_version) # h5py already reports this
    print("Keras:", str(keras.__version__))
    print("Tensorflow:", str(tf.__version__))

    if list_devices:
        from tensorflow.python.client import device_lib
        print("Devices:\n" + str(device_lib.list_local_devices()))


def find_weights(model_path):
    """ Returns paths to saved weights in the run's subfolder.  """
    weights_folder = os.path.join(model_path, "weights")
    weights_paths = sorted(os.listdir(weights_folder))
    weights_paths = [x for x in weights_paths if "weights" in x]
    matches = [re.match("weights[.]([0-9]+)-([0-9.]+)[.]h5", x).groups() for x in weights_paths]
    epochs = np.array([int(x[0]) for x in matches])
    val_losses = np.array([np.float(x[1]) for x in matches])
    
    weights_paths = [os.path.join(weights_folder, x) for x in weights_paths]
    return weights_paths, epochs, val_losses


def find_best_weights(model_path):
    """ Returns the path to the model weights with the lowest validation loss. """
    weights_paths, epochs, val_losses = find_weights(model_path)
    if len(val_losses) > 0:
        idx = np.argmin(val_losses)
        return weights_paths[idx]
    else:
        return None


def load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0,3,2,1)):
    """ Loads and normalizes datasets. """
    
    # Load
    t0 = time()
    with h5py.File(data_path,"r") as f:
        X = f[X_dset][:]
        Y = f[Y_dset][:]
    print("Loaded %d samples [%.1fs]" % (len(X), time() - t0))
    
    # Adjust dimensions
    t0 = time()
    X = preprocess(X, permute)
    Y = preprocess(Y, permute)
    print("Permuted and normalized data. [%.1fs]" % (time() - t0))
    
    return X, Y

def preprocess(X, permute=(0,3,2,1)):
    """ Normalizes input data. """
    
    # Add singleton dim for single images
    if X.ndim == 3:
        X = X[None,...]
    
    # Adjust dimensions
    X = np.transpose(X, permute)
    
    # Normalize
    if X.dtype == "uint8":
        X = X.astype("float32") / 255
    
    return X

def load_video(data_path, X_dset="box", permute=(0,3,2,1)):
    """ Loads and normalizes videos. """
    
    # Load
    t0 = time()
    with h5py.File(data_path,"r") as f:
        X = f[X_dset][:]
    print("Loaded %d samples [%.1fs]" % (len(X), time() - t0))
    print("Image samples shape: ", X.shape)
    
    # Adjust dimensions
    t0 = time()
    X = preprocess(X, permute)
    print("Permuted and normalized data. [%.1fs]" % (time() - t0))
    
    return X

def load_label(label_path, number_of_samples, rows, cols, channels=1, permute=None):
    """ Loads label and generate confidence maps"""

    # Load
    t0 = time()
    point = np.zeros((number_of_samples, 2, channels))
    for i in range(len(label_path)):
        with open(label_path[i], 'r') as f:
            for line in f.readlines():
                # frame_idx, x, y, w, h, conf
                line = line.split(" ")[:-1]
                assert len(line) == 5
                frame_idx = int(line[0])
                x = int(line[1]) + int(line[3]) // 2
                y = int(line[2]) + int(line[4]) // 2
                point[frame_idx - 1:frame_idx, :, i] = np.array((y, x))

    start_time = time()
    confmap = px2confmap(point, number_of_samples, rows, cols, channels=channels)
    print("Generate conf map time: {}s".format(time() - start_time))

    if permute is not None:
        confmap = preprocess(confmap, permute)

    return confmap

def px2confmap(point, number_of_samples, rows, cols, channels=1, sigma=5, normalize=True):
    assert channels >= 1
    XX = np.arange(rows * cols).reshape(rows, cols, 1) // cols
    YY = np.arange(rows * cols).reshape(rows, cols, 1) % cols
    XX = np.concatenate([XX for i in range(channels)], axis=-1)
    YY = np.concatenate([YY for i in range(channels)], axis=-1)
    x = point[:, 0, :].reshape(number_of_samples, 1, 1, channels)
    y = point[:, 1, :].reshape(number_of_samples, 1, 1, channels)
    confmap = np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / 2 / (sigma ** 2))
    """for i in range(number_of_samples):
        x = point[i, 0, 0]
        y = point[i, 1, 0]
        confmap[i:i+1, :, :, 0] = np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / 2 / (sigma ** 2))
        Yi = confmap[i:i+1, :, :, 0]"""

    if not normalize:
        confmap /= (sigma * math.sqrt(2 * math.pi))
    
    return confmap