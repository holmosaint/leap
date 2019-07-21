import numpy as np
import h5py
import os
from time import time
from scipy.io import loadmat, savemat
import re
import shutil
import clize
import pandas as pd

import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

from leap import models
from leap.image_augmentation import PairedImageAugmenter, MultiInputOutputPairedImageAugmenter
from leap.viz import show_pred, show_confmap_grid, plot_history
from leap.utils import load_dataset, load_video, load_label, load_confmap

import argparse
import sys
import scipy.io as sio
import math


def train_val_split(X, Y, val_size=0.15, shuffle=True):
    """ Splits datasets into training and validation sets. """

    if val_size < 1:
        val_size = int(np.round(len(X) * val_size))

    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)

    val_idx = idx[:val_size]
    idx = idx[val_size:]

    return X[idx], Y[idx], X[val_idx], Y[val_idx], idx, val_idx

def train_val_test_split(X, Y, train_size=0.5, val_size=0.15, test_size=None, shuffle=True):
    """ Splits datasets into training and validation sets """
    """ The test sequence should be sequential """

    if train_size < 1:
        train_size = int(np.round(len(X) * train_size))
    if val_size < 1:
        val_size = int(train_size * val_size)
    if test_size is not None:
        if test_size < 1:
            test_size = int(np.round(len(X) * test_size))
    else:
        test_size = len(X) - train_size
    
    test_start = np.random.randint(0, len(X) - test_size - 1)
    test_idx = np.arange(test_start, test_start + test_size)

    train_set = list(range(0, test_start))
    train_set.extend(list(range(test_start + test_size, len(X))))
    train_set = np.array(train_set)
    if shuffle:
        np.random.shuffle(train_set)

    val_idx = train_set[:val_size]
    train_idx = train_set[val_size:train_size]
    # test_idx = idx[train_size:]

    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], X[test_idx], Y[test_idx], train_idx, val_idx, test_idx



def create_run_folders(run_name, base_path="models", clean=False):
    """ Creates subfolders necessary for outputs of training. """

    def is_empty_run(run_path):
        weights_path = os.path.join(run_path, "weights")
        has_weights_folder = os.path.exists(weights_path)
        return not has_weights_folder or len(os.listdir(weights_path)) == 0

    run_path = os.path.join(base_path, run_name)

    if not clean:
        initial_run_path = run_path
        i = 1
        while os.path.exists(run_path): #and not is_empty_run(run_path):
            run_path = "%s_%02d" % (initial_run_path, i)
            i += 1

    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    os.makedirs(run_path)
    os.makedirs(os.path.join(run_path, "weights"))
    os.makedirs(os.path.join(run_path, "viz_pred"))
    os.makedirs(os.path.join(run_path, "viz_confmaps"))
    print("Created folder:", run_path)

    return run_path



class LossHistory(keras.callbacks.Callback):
    def __init__(self, run_path):
        super().__init__()
        self.run_path = run_path

    def on_train_begin(self, logs={}):
        self.history = []

    def on_epoch_end(self, batch, logs={}):
        # Append to log list
        self.history.append(logs.copy())

        # Save history so far to MAT file
        savemat(os.path.join(self.run_path, "history.mat"),
                {k: [x[k] for x in self.history] for k in self.history[0].keys()})

        # Plot graph
        plot_history(self.history, save_path=os.path.join(self.run_path, "history.png"))


def create_model(net_name, img_size, output_channels, **kwargs):
    """ Wrapper for initializing a network for training. """
    # compile_model = getattr(models, net_name)

    compile_model = dict(
        leap_cnn=models.leap_cnn,
        hourglass=models.hourglass,
        stacked_hourglass=models.stacked_hourglass,
        ).get(net_name)
    if compile_model == None:
        return None

    return compile_model(img_size, output_channels, **kwargs)

def train_test_same(data_path, label_path, *,
    base_output_path="models",
    run_name=None,
    data_name=None,
    net_name="leap_cnn",
    clean=False,
    box_dset="box",
    confmap_dset="confmaps",
    train_size=800,
    val_size=0.1,
    preshuffle=True,
    filters=64,
    rotate_angle=15,
    epochs=20,
    batch_size=8,
    batches_per_epoch=90,
    val_batches_per_epoch=10,
    viz_idx=[0, 100, 200, 300, 400],
    reduce_lr_factor=0.1,
    reduce_lr_patience=3,
    reduce_lr_min_delta=1e-5,
    reduce_lr_cooldown=0,
    reduce_lr_min_lr=1e-10,
    save_every_epoch=False,
    amsgrad=False,
    upsampling_layers=True,
    ):
    """
    Trains the network and saves the intermediate results to an output directory.

    :param data_path: Path to an HDF5 file with box and confmaps datasets
    :param base_output_path: Path to folder in which the run data folder will be saved
    :param run_name: Name of the training run. If not specified, will be formatted according to other parameters.
    :param data_name: Name of the dataset for use in formatting run_name
    :param net_name: Name of the network for use in formatting run_name
    :param clean: If True, deletes the contents of the run output path
    :param box_dset: Name of the box dataset in the HDF5 data file
    :param confmap_dset: Name of the confidence maps dataset in the HDF5 data file
    :param preshuffle: If True, shuffle prior to splitting the dataset, otherwise validation set will be the last frames
    :param val_size: Fraction of dataset to use as validation
    :param filters: Number of filters to use as baseline (see create_model)
    :param rotate_angle: Images will be augmented by rotating by +-rotate_angle
    :param epochs: Number of epochs to train for
    :param batch_size: Number of samples per batch
    :param batches_per_epoch: Number of batches per epoch (validation is evaluated at the end of the epoch)
    :param val_batches_per_epoch: Number of batches for validation
    :param viz_idx: Index of the sample image to use for visualization
    :param reduce_lr_factor: Factor to reduce the learning rate by (see ReduceLROnPlateau)
    :param reduce_lr_patience: How many epochs to wait before reduction (see ReduceLROnPlateau)
    :param reduce_lr_min_delta: Minimum change in error required before reducing LR (see ReduceLROnPlateau)
    :param reduce_lr_cooldown: How many epochs to wait after reduction before LR can be reduced again (see ReduceLROnPlateau)
    :param reduce_lr_min_lr: Minimum that the LR can be reduced down to (see ReduceLROnPlateau)
    :param save_every_epoch: Save weights at every epoch. If False, saves only initial, final and best weights.
    :param amsgrad: Use AMSGrad variant of optimizer. Can help with training accuracy on rare examples (see Reddi et al., 2018)
    :param upsampling_layers: Use simple bilinear upsampling layers as opposed to learned transposed convolutions
    """
    batches_per_epoch = math.ceil(train_size / batch_size)

    if len(data_path) != len(label_path):
        print("The size of data path is not the same as label path")
        sys.exit(1)

    # Load
    print("data_path:", data_path)
    # box, confmap = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset)
    box = list()
    for i in range(len(data_path)):
        box.append(load_video(data_path[i], X_dset=box_dset))
        print("Data set {} shape: {}".format(i, box[i].shape))

    # confmap = load_label(label_path, *box.shape[:-1], channels=len(label_path))
    confmap = list()
    for i in range(len(label_path)):
        confmap.append(load_confmap(label_path[i]))
    # viz_sample = (box[viz_idx], confmap[viz_idx])
    viz_sample = [(box[0][x], confmap[0][x]) for x in viz_idx]

    # Pull out metadata
    img_size = box[0].shape[1:]
    num_output_channels = confmap[0].shape[-1]
    print("img_size:", img_size)
    print("num_output_channels:", num_output_channels)

    # Build run name if needed
    if data_name == None:
        data_name = os.path.splitext(os.path.basename(data_path[0]))[0]
    if run_name == None:
        # Ex: "WangMice-DiegoCNN_v1.0_filters=64_rot=15_lrfactor=0.1_lrmindelta=1e-05"
        # run_name = "%s-%s_filters=%d_rot=%d_lrfactor=%.1f_lrmindelta=%g" % (data_name, net_name, filters, rotate_angle, reduce_lr_factor, reduce_lr_min_delta)
        run_name = "%s-%s_epochs=%d" % (data_name, net_name, epochs)
    print("data_name:", data_name)
    print("run_name:", run_name)

    # Create network
    if isinstance(net_name, keras.models.Model):
        model = net_name
        net_name = model.name
    else:
        model = create_model(net_name, img_size, num_output_channels, filters=filters, amsgrad=amsgrad, upsampling_layers=upsampling_layers, summary=True)
    if model == None:
        print("Could not find model:", net_name)
        return

    # box = list()
    train_box = list()
    val_box = list()
    test_box = list()
    # confmap = list()
    train_confmap = list()
    val_confmap = list()
    test_confmap = list()
    train_idx = list()
    val_idx = list()
    test_idx = list()
    for i in range(len(data_path)):
        b, cm, vb, vcm, tb, tcm, tid, vid, teid \
            = train_val_test_split(box[i], confmap[i], train_size=train_size, val_size=val_size, test_size=2500)
        train_box.append(b)
        train_confmap.append(cm)
        val_box.append(vb)
        val_confmap.append(vcm)
        test_box.append(tb)
        test_confmap.append(tcm)
        train_idx.append(tid)
        val_idx.append(vid)
        test_idx.append(teid)
    box = train_box
    confmap = train_confmap

    # model.load_weights("/home/retina/skw/work/leap/leap/models/video1-leap_cnn_epochs=10_04/final_model.h5")
    # Initialize run directories
    run_path = create_run_folders(run_name, base_path=base_output_path, clean=clean)
    savemat(os.path.join(run_path, "training_info.mat"),
            {"data_path": data_path, "test_idx": test_idx, "val_idx": val_idx, "train_idx": train_idx,
             "base_output_path": base_output_path, "run_name": run_name, "data_name": data_name,
             "net_name": net_name, "clean": clean, "box_dset": box_dset, "confmap_dset": confmap_dset,
             "preshuffle": preshuffle, "val_size": val_size, "filters": filters, "rotate_angle": rotate_angle,
             "epochs": epochs, "batch_size": batch_size, "batches_per_epoch": batches_per_epoch,
             "val_batches_per_epoch": val_batches_per_epoch, "viz_idx": viz_idx, "reduce_lr_factor": reduce_lr_factor,
             "reduce_lr_patience": reduce_lr_patience, "reduce_lr_min_delta": reduce_lr_min_delta,
             "reduce_lr_cooldown": reduce_lr_cooldown, "reduce_lr_min_lr": reduce_lr_min_lr,
             "save_every_epoch": save_every_epoch, "amsgrad": amsgrad, "upsampling_layers": upsampling_layers})

    # Save initial network
    model.save(os.path.join(run_path, "initial_model.h5"))

    # Data generators/augmentation
    input_layers = model.input_names
    output_layers = model.output_names
    box = np.concatenate(box, axis=0)
    val_box = np.concatenate(val_box, axis=0)
    # test_box = np.concatenate(test_box, axis=0)
    confmap = np.concatenate(confmap, axis=0)
    val_confmap = np.concatenate(val_confmap, axis=0)
    # test_confmap = np.concatenate(test_confmap, axis=0)
    if len(input_layers) > 1 or len(output_layers) > 1:
        train_datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, box, confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        val_datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, val_box, val_confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        test_datagen = list()
        for i in range(len(data_path)):
            test_datagen.append(MultiInputOutputPairedImageAugmenter(input_layers, output_layers, test_box[i], test_confmap[i], batch_size=batch_size, shuffle=False, theta=(-rotate_angle, rotate_angle)))
    else:
        train_datagen = PairedImageAugmenter(box, confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        val_datagen = PairedImageAugmenter(val_box, val_confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        test_datagen = list()
        for i in range(len(data_path)):
            test_datagen.append(PairedImageAugmenter(test_box[i], test_confmap[i], batch_size=batch_size, shuffle=False, theta=(-rotate_angle, rotate_angle)))

    # Initialize training callbacks
    history_callback = LossHistory(run_path=run_path)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_factor,
                                          patience=reduce_lr_patience, verbose=1, mode="auto",
                                          epsilon=reduce_lr_min_delta, cooldown=reduce_lr_cooldown,
                                          min_lr=reduce_lr_min_lr)
    if save_every_epoch:
        checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.h5"), verbose=1, save_best_only=False)
    else:
        checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.h5"), verbose=1, save_best_only=True)
    viz_grid_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_confmap_grid(model, viz_sample, plot=True, save_path=os.path.join(run_path, "viz_confmaps/confmaps_%03d.png" % epoch), show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(model, viz_sample, save_path=os.path.join(run_path, "viz_pred/pred_%03d.png" % epoch), show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: show_pred(model, viz_sample, save_path=os.path.join(run_path, "viz_pred/pred_%03d.png" % epoch), show_figure=False))

    # release memory
    train_box.clear()
    val_box = 0
    test_box = 0
    train_confmap.clear()
    val_confmap = 0
    # test_confmap = list()
    train_idx = 0
    val_idx = 0
    test_idx = 0
    box = 0
    confmap = 0

    # Train!
    epoch0 = 0
    t0_train = time()
    training = model.fit_generator(
            train_datagen,
            initial_epoch=epoch0,
            epochs=epochs,
            verbose=1,
    #         use_multiprocessing=True,
    #         workers=8,
            steps_per_epoch=batches_per_epoch,
            max_queue_size=512,
            shuffle=False,
            validation_data=val_datagen,
            validation_steps=val_batches_per_epoch,
            callbacks = [
                reduce_lr_callback,
                checkpointer,
                history_callback,
                viz_pred_callback,
                viz_grid_callback
            ]
        )

    # Compute total elapsed time for training
    elapsed_train = time() - t0_train
    print("Total runtime: %.1f mins" % (elapsed_train / 60))

    # Save final model
    model.history = history_callback.history
    model.save(os.path.join(run_path, "final_model.h5"))
    print("Model saved as ", os.path.join(run_path, "final_model.h5"))
    
    print("Checking accuracy...")
    for i in range(len(data_path)):
        """ evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0) """
        t0_test = time()
        evaluation = model.predict_generator(test_datagen[i], steps=None, max_queue_size=10, workers=4, use_multiprocessing=False, verbose=1)
        elapsed_test = time() - t0_test
        print("Total evaluation time on dataset %d: %.1f mins" % (i, elapsed_test / 60))

        L = ['back_foot', 'front_foot']
        for j in range(num_output_channels):
            Yi = test_confmap[i][:, :, :, j]
            Y = Yi.reshape(Yi.shape[0], -1)
            print(Y.shape)
            Y = np.argmax(Y, axis=-1)
            gt_coord = np.unravel_index(Y.reshape(2500, -1), Yi.shape)
            gt_coord = np.concatenate(gt_coord, axis=-1)
            print(gt_coord.shape)
            
            Yi = evaluation[:, :, :, j]
            Y = Yi.reshape(Yi.shape[0], -1)
            print(Y.shape)
            Y = np.argmax(Y, axis=-1)
            evaluation_coord = np.unravel_index(Y.reshape(2500, -1), Yi.shape)
            evaluation_coord = np.concatenate(evaluation_coord, axis=-1)

            res = np.linalg.norm(gt_coord - evaluation_coord, axis=-1, keepdims=True)
            print(res.shape)
            res = res < 20
            res = res.astype(np.int32)
            np.savetxt(os.path.join(run_path, "dataset_" + str(i) + "_" + L[j] + "_test_result.txt"), res.reshape(-1, 1))
            res = np.mean(res)
            print("Accuracy on data set %d: %.2f of " % (i, res), L[j])

def train_test_diff(data_path, label_path, test_data_path, test_label_path, *,
    base_output_path="models",
    run_name=None,
    data_name=None,
    net_name="leap_cnn",
    clean=False,
    box_dset="box",
    confmap_dset="confmaps",
    train_size=800,
    val_size=0.1,
    preshuffle=True,
    filters=64,
    rotate_angle=15,
    epochs=20,
    batch_size=8,
    batches_per_epoch=90,
    val_batches_per_epoch=10,
    viz_idx=[0, 100, 200, 300, 400],
    reduce_lr_factor=0.1,
    reduce_lr_patience=3,
    reduce_lr_min_delta=1e-5,
    reduce_lr_cooldown=0,
    reduce_lr_min_lr=1e-10,
    save_every_epoch=False,
    amsgrad=False,
    upsampling_layers=True,
    ):
    """
    Trains the network and saves the intermediate results to an output directory.

    :param data_path: Path to an HDF5 file with box and confmaps datasets
    :param base_output_path: Path to folder in which the run data folder will be saved
    :param run_name: Name of the training run. If not specified, will be formatted according to other parameters.
    :param data_name: Name of the dataset for use in formatting run_name
    :param net_name: Name of the network for use in formatting run_name
    :param clean: If True, deletes the contents of the run output path
    :param box_dset: Name of the box dataset in the HDF5 data file
    :param confmap_dset: Name of the confidence maps dataset in the HDF5 data file
    :param preshuffle: If True, shuffle prior to splitting the dataset, otherwise validation set will be the last frames
    :param val_size: Fraction of dataset to use as validation
    :param filters: Number of filters to use as baseline (see create_model)
    :param rotate_angle: Images will be augmented by rotating by +-rotate_angle
    :param epochs: Number of epochs to train for
    :param batch_size: Number of samples per batch
    :param batches_per_epoch: Number of batches per epoch (validation is evaluated at the end of the epoch)
    :param val_batches_per_epoch: Number of batches for validation
    :param viz_idx: Index of the sample image to use for visualization
    :param reduce_lr_factor: Factor to reduce the learning rate by (see ReduceLROnPlateau)
    :param reduce_lr_patience: How many epochs to wait before reduction (see ReduceLROnPlateau)
    :param reduce_lr_min_delta: Minimum change in error required before reducing LR (see ReduceLROnPlateau)
    :param reduce_lr_cooldown: How many epochs to wait after reduction before LR can be reduced again (see ReduceLROnPlateau)
    :param reduce_lr_min_lr: Minimum that the LR can be reduced down to (see ReduceLROnPlateau)
    :param save_every_epoch: Save weights at every epoch. If False, saves only initial, final and best weights.
    :param amsgrad: Use AMSGrad variant of optimizer. Can help with training accuracy on rare examples (see Reddi et al., 2018)
    :param upsampling_layers: Use simple bilinear upsampling layers as opposed to learned transposed convolutions
    """
    batches_per_epoch = math.ceil(train_size / batch_size)

    if len(data_path) != len(label_path):
        print("The size of data path of training data is not the same as label path")
        sys.exit(1)

    if len(test_data_path) != len(test_label_path):
        print("The size of data path of testing data is not the same as label path")
        sys.exit(1)


    # Load
    print("Loading training data...")
    print("data_path:", data_path)
    # box, confmap = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset)
    box = list()
    for i in range(len(data_path)):
        box.append(load_video(data_path[i], X_dset=box_dset))
        print("Data set {} shape: {}".format(i, box[i].shape))
    print("Length of box: ", len(box))

    # confmap = load_label(label_path, *box.shape[:-1], channels=len(label_path))
    confmap = list()
    for i in range(len(label_path)):
        confmap.append(load_confmap(label_path[i]))
    print("Length of confmap: ", len(confmap))

    # viz_sample = (box[viz_idx], confmap[viz_idx])
    viz_sample = [(box[0][x], confmap[0][x]) for x in viz_idx]

    # Pull out metadata
    img_size = box[0].shape[1:]
    num_output_channels = confmap[0].shape[-1]
    print("img_size:", img_size)
    print("num_output_channels:", num_output_channels)

    # Build run name if needed
    if data_name == None:
        data_name = os.path.splitext(os.path.basename(data_path[0]))[0]
    if run_name == None:
        # Ex: "WangMice-DiegoCNN_v1.0_filters=64_rot=15_lrfactor=0.1_lrmindelta=1e-05"
        # run_name = "%s-%s_filters=%d_rot=%d_lrfactor=%.1f_lrmindelta=%g" % (data_name, net_name, filters, rotate_angle, reduce_lr_factor, reduce_lr_min_delta)
        run_name = "%s-%s_epochs=%d" % (data_name, net_name, epochs)
    print("data_name:", data_name)
    print("run_name:", run_name)

    # Create network
    if isinstance(net_name, keras.models.Model):
        model = net_name
        net_name = model.name
    else:
        model = create_model(net_name, img_size, num_output_channels, filters=filters, amsgrad=amsgrad, upsampling_layers=upsampling_layers, summary=True)
    if model == None:
        print("Could not find model:", net_name)
        return

    # box = list()
    train_box = list()
    val_box = list()
    # test_box = list()
    # confmap = list()
    train_confmap = list()
    val_confmap = list()
    test_confmap = list()
    train_idx = list()
    val_idx = list()
    test_idx = list()
    for i in range(len(data_path)):
        b, cm, vb, vcm, tb, tcm, tid, vid, teid \
            = train_val_test_split(box[i], confmap[i], train_size=train_size, val_size=val_size, test_size=2500)
        train_box.append(b)
        train_confmap.append(cm)
        val_box.append(vb)
        val_confmap.append(vcm)
        # test_box.append(tb)
        # test_confmap.append(tcm)
        train_idx.append(tid)
        val_idx.append(vid)
        # test_idx.append(teid)
    box = train_box
    confmap = train_confmap

    # Load
    print("Loading testing data...")
    print("data_path:", test_data_path)
    test_box = list()
    for i in range(len(test_data_path)):
        test_box.append(load_video(test_data_path[i], X_dset=box_dset))
        print("Data set {} shape: {}".format(i, test_box[i].shape))

    # confmap = load_label(label_path, *box.shape[:-1], channels=len(label_path))
    test_confmap = list()
    for i in range(len(test_label_path)):
        test_confmap.append(load_confmap(test_label_path[i]))

    # model.load_weights("/home/retina/skw/work/leap/leap/models/video1-leap_cnn_epochs=10_04/final_model.h5")
    # Initialize run directories
    run_path = create_run_folders(run_name, base_path=base_output_path, clean=clean)
    savemat(os.path.join(run_path, "training_info.mat"),
            {"data_path": data_path, "test_idx": test_idx, "val_idx": val_idx, "train_idx": train_idx,
             "base_output_path": base_output_path, "run_name": run_name, "data_name": data_name,
             "net_name": net_name, "clean": clean, "box_dset": box_dset, "confmap_dset": confmap_dset,
             "preshuffle": preshuffle, "val_size": val_size, "filters": filters, "rotate_angle": rotate_angle,
             "epochs": epochs, "batch_size": batch_size, "batches_per_epoch": batches_per_epoch,
             "val_batches_per_epoch": val_batches_per_epoch, "viz_idx": viz_idx, "reduce_lr_factor": reduce_lr_factor,
             "reduce_lr_patience": reduce_lr_patience, "reduce_lr_min_delta": reduce_lr_min_delta,
             "reduce_lr_cooldown": reduce_lr_cooldown, "reduce_lr_min_lr": reduce_lr_min_lr,
             "save_every_epoch": save_every_epoch, "amsgrad": amsgrad, "upsampling_layers": upsampling_layers})

    # Save initial network
    model.save(os.path.join(run_path, "initial_model.h5"))

    # Data generators/augmentation
    input_layers = model.input_names
    output_layers = model.output_names
    box = np.concatenate(box, axis=0)
    val_box = np.concatenate(val_box, axis=0)
    # test_box = np.concatenate(test_box, axis=0)
    confmap = np.concatenate(confmap, axis=0)
    val_confmap = np.concatenate(val_confmap, axis=0)
    # test_confmap = np.concatenate(test_confmap, axis=0)
    if len(input_layers) > 1 or len(output_layers) > 1:
        train_datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, box, confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        val_datagen = MultiInputOutputPairedImageAugmenter(input_layers, output_layers, val_box, val_confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        test_datagen = list()
        for i in range(len(data_path)):
            test_datagen.append(MultiInputOutputPairedImageAugmenter(input_layers, output_layers, test_box[i], test_confmap[i], batch_size=batch_size, shuffle=False, theta=(-rotate_angle, rotate_angle)))
    else:
        train_datagen = PairedImageAugmenter(box, confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        val_datagen = PairedImageAugmenter(val_box, val_confmap, batch_size=batch_size, shuffle=True, theta=(-rotate_angle, rotate_angle))
        test_datagen = list()
        for i in range(len(data_path)):
            test_datagen.append(PairedImageAugmenter(test_box[i], test_confmap[i], batch_size=batch_size, shuffle=False, theta=(-rotate_angle, rotate_angle)))

    # Initialize training callbacks
    history_callback = LossHistory(run_path=run_path)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_factor,
                                          patience=reduce_lr_patience, verbose=1, mode="auto",
                                          epsilon=reduce_lr_min_delta, cooldown=reduce_lr_cooldown,
                                          min_lr=reduce_lr_min_lr)
    if save_every_epoch:
        checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "weights/weights.{epoch:03d}-{val_loss:.9f}.h5"), verbose=1, save_best_only=False)
    else:
        checkpointer = ModelCheckpoint(filepath=os.path.join(run_path, "best_model.h5"), verbose=1, save_best_only=True)
    viz_grid_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_confmap_grid(model, viz_sample, plot=True, save_path=os.path.join(run_path, "viz_confmaps/confmaps_%03d.png" % epoch), show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: show_pred(model, viz_sample, save_path=os.path.join(run_path, "viz_pred/pred_%03d.png" % epoch), show_figure=False))
    viz_pred_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: show_pred(model, viz_sample, save_path=os.path.join(run_path, "viz_pred/pred_%03d.png" % epoch), show_figure=False))

    # Train!
    epoch0 = 0
    t0_train = time()
    training = model.fit_generator(
            train_datagen,
            initial_epoch=epoch0,
            epochs=epochs,
            verbose=1,
    #         use_multiprocessing=True,
    #         workers=8,
            steps_per_epoch=batches_per_epoch,
            max_queue_size=512,
            shuffle=False,
            validation_data=val_datagen,
            validation_steps=val_batches_per_epoch,
            callbacks = [
                reduce_lr_callback,
                checkpointer,
                history_callback,
                viz_pred_callback,
                viz_grid_callback
            ]
        )

    # Compute total elapsed time for training
    elapsed_train = time() - t0_train
    print("Total runtime: %.1f mins" % (elapsed_train / 60))

    # Save final model
    model.history = history_callback.history
    model.save(os.path.join(run_path, "final_model.h5"))
    print("Model saved as ", os.path.join(run_path, "final_model.h5"))
    
    print("Checking accuracy...")
    for i in range(len(data_path)):
        """ evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0) """
        t0_test = time()
        evaluation = model.predict_generator(test_datagen[i], steps=None, max_queue_size=10, workers=4, use_multiprocessing=False, verbose=1)
        elapsed_test = time() - t0_test
        print("Total evaluation time on dataset %d: %.1f mins" % (i, elapsed_test / 60))

        Yi = test_confmap[i]
        Y = Yi.reshape(Yi.shape[0], -1)
        print(Y.shape)
        Y = np.argmax(Y, axis=-1)
        gt_coord = np.unravel_index(Y.reshape(2500, -1), Yi.shape)
        gt_coord = np.concatenate(gt_coord, axis=-1)
        print(gt_coord.shape)
        
        Yi = evaluation
        Y = Yi.reshape(Yi.shape[0], -1)
        print(Y.shape)
        Y = np.argmax(Y, axis=-1)
        evaluation_coord = np.unravel_index(Y.reshape(2500, -1), Yi.shape)
        evaluation_coord = np.concatenate(evaluation_coord, axis=-1)

        res = np.linalg.norm(gt_coord - evaluation_coord, axis=-1, keepdims=True)
        print(res.shape)
        res = res < 20
        res = res.astype(np.int32)
        np.savetxt(os.path.join(run_path, "dataset_" + str(i) + "_" + L[j] + "_test_result.txt"), res.reshape(-1, 1))
        res = np.mean(res)
        print("Accuracy on data set %d: %.2f" % (i, res))

def test(data_path, label_path, model_path, test_idx, *,
    base_output_path="models",
    run_name=None,
    data_name=None,
    net_name="leap_cnn",
    clean=False,
    box_dset="box",
    confmap_dset="confmaps",
    train_size=800,
    val_size=0.15,
    preshuffle=True,
    filters=64,
    rotate_angle=15,
    epochs=10,
    batch_size=10,
    batches_per_epoch=320,
    val_batches_per_epoch=10,
    viz_idx=[0, 100, 200, 300, 400],
    reduce_lr_factor=0.1,
    reduce_lr_patience=3,
    reduce_lr_min_delta=1e-5,
    reduce_lr_cooldown=0,
    reduce_lr_min_lr=1e-10,
    save_every_epoch=False,
    amsgrad=False,
    upsampling_layers=True,
    ):
    """
    Trains the network and saves the intermediate results to an output directory.

    :param data_path: Path to an HDF5 file with box and confmaps datasets
    :param base_output_path: Path to folder in which the run data folder will be saved
    :param run_name: Name of the training run. If not specified, will be formatted according to other parameters.
    :param data_name: Name of the dataset for use in formatting run_name
    :param net_name: Name of the network for use in formatting run_name
    :param clean: If True, deletes the contents of the run output path
    :param box_dset: Name of the box dataset in the HDF5 data file
    :param confmap_dset: Name of the confidence maps dataset in the HDF5 data file
    :param preshuffle: If True, shuffle prior to splitting the dataset, otherwise validation set will be the last frames
    :param val_size: Fraction of dataset to use as validation
    :param filters: Number of filters to use as baseline (see create_model)
    :param rotate_angle: Images will be augmented by rotating by +-rotate_angle
    :param epochs: Number of epochs to train for
    :param batch_size: Number of samples per batch
    :param batches_per_epoch: Number of batches per epoch (validation is evaluated at the end of the epoch)
    :param val_batches_per_epoch: Number of batches for validation
    :param viz_idx: Index of the sample image to use for visualization
    :param reduce_lr_factor: Factor to reduce the learning rate by (see ReduceLROnPlateau)
    :param reduce_lr_patience: How many epochs to wait before reduction (see ReduceLROnPlateau)
    :param reduce_lr_min_delta: Minimum change in error required before reducing LR (see ReduceLROnPlateau)
    :param reduce_lr_cooldown: How many epochs to wait after reduction before LR can be reduced again (see ReduceLROnPlateau)
    :param reduce_lr_min_lr: Minimum that the LR can be reduced down to (see ReduceLROnPlateau)
    :param save_every_epoch: Save weights at every epoch. If False, saves only initial, final and best weights.
    :param amsgrad: Use AMSGrad variant of optimizer. Can help with training accuracy on rare examples (see Reddi et al., 2018)
    :param upsampling_layers: Use simple bilinear upsampling layers as opposed to learned transposed convolutions
    """
    if len(data_path) > 1 or len(label_path) > 1:
        print("Test only supports single date set yet")
        sys.exit(1)

    if len(data_path) != len(label_path):
        print("The size of data path is not the same as label path")
        sys.exit(1)

    # Load
    print("data_path:", data_path)
    # box, confmap = load_dataset(data_path, X_dset=box_dset, Y_dset=confmap_dset)
    box = list()
    for i in range(len(data_path)):
        box.append(load_video(data_path[0], X_dset=box_dset))
        print("Data set {} shape: {}".format(i, box[i].shape))

    # confmap = load_label(label_path, *box.shape[:-1], channels=len(label_path))
    confmap = list()
    for i in range(len(label_path)):
        confmap.append(load_confmap(label_path[i]))

    # Pull out metadata
    img_size = box[0].shape[1:-1]
    num_output_channels = confmap[0].shape[-1]
    print("img_size:", img_size)
    print("num_output_channels:", num_output_channels)

    if type(test_idx) == str:
        mat_contends = sio.loadmat(test_idx)
        test_idx = mat_contends["test_idx"]
    
    print(type(test_idx))
    print(test_idx.shape)

    box[0] = box[0][test_idx][0]
    confmap[0] = confmap[0][test_idx][0]

    # Build run name if needed
    if data_name == None:
        data_name = os.path.splitext(os.path.basename(data_path[0]))[0]
    if run_name == None:
        # Ex: "WangMice-DiegoCNN_v1.0_filters=64_rot=15_lrfactor=0.1_lrmindelta=1e-05"
        # run_name = "%s-%s_filters=%d_rot=%d_lrfactor=%.1f_lrmindelta=%g" % (data_name, net_name, filters, rotate_angle, reduce_lr_factor, reduce_lr_min_delta)
        run_name = "%s-%s_epochs=%d" % (data_name, net_name, epochs)
    print("data_name:", data_name)
    print("run_name:", run_name)

    # Create network
    if isinstance(net_name, keras.models.Model):
        model = net_name
        net_name = model.name
    else:
        model = create_model(net_name, img_size, num_output_channels, filters=filters, amsgrad=amsgrad, upsampling_layers=upsampling_layers, summary=True)
    if model == None:
        print("Could not find model:", net_name)
        return

    model.load_weights(model_path)
    # Initialize run directories
    run_path = create_run_folders(run_name, base_path=base_output_path, clean=clean)

    # Data generators/augmentation
    input_layers = model.input_names
    output_layers = model.output_names
    if len(input_layers) > 1 or len(output_layers) > 1:
        test_datagen = list()
        for i in range(len(data_path)):
            test_datagen.append(MultiInputOutputPairedImageAugmenter(input_layers, output_layers, box[0], confmap[0], batch_size=batch_size, shuffle=False, theta=(-rotate_angle, rotate_angle)))
    else:
        test_datagen = list()
        for i in range(len(data_path)):
            test_datagen.append(PairedImageAugmenter(box[0], confmap[0], batch_size=batch_size, shuffle=False, theta=(-rotate_angle, rotate_angle)))

    # release memory
    box.clear()

    test_len = test_idx.shape[-1]

    print("Checking accuracy...")
    for i in range(len(data_path)):
        """ evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0) """
        t0_test = time()
        evaluation = model.predict_generator(test_datagen[i], steps=None, max_queue_size=10, workers=16, use_multiprocessing=True, verbose=1)
        elapsed_test = time() - t0_test
        print("Total evaluation time on dataset %d: %.1f mins" % (i, elapsed_test / 60))

        L = ['back_foot', 'front_foot']
        for j in range(num_output_channels):
            Yi = confmap[i][:, :, :, j]
            Y = Yi.reshape(Yi.shape[0], -1)
            print(Y.shape)
            Y = np.argmax(Y, axis=-1)
            gt_coord = np.unravel_index(Y.reshape(test_len, -1), Yi.shape)
            gt_coord = np.concatenate(gt_coord, axis=-1)
            print(gt_coord.shape)
            
            Yi = evaluation[:, :, :, j]
            Y = Yi.reshape(Yi.shape[0], -1)
            print(Y.shape)
            Y = np.argmax(Y, axis=-1)
            evaluation_coord = np.unravel_index(Y.reshape(test_len, -1), Yi.shape)
            evaluation_coord = np.concatenate(evaluation_coord, axis=-1)

            res = np.linalg.norm(gt_coord - evaluation_coord, axis=-1, keepdims=True)
            print(res.shape)
            res = res < 20
            res = res.astype(np.int32)
            np.savetxt(os.path.join(run_path, "dataset_" + str(i) + "_" + L[j] + "_test_result.txt"), res.reshape(-1, 1))
            res = np.mean(res)
            print("Accuracy on data set %d: %.2f of " % (i, res), L[j])
            print("Save results in ", os.path.join(run_path, "dataset_" + str(i) + "_" + L[j] + "_test_result.txt"))

def cal_acc(label_path=None):
    confmap = load_label(label_path, *(5000, 600, 896, 1))
    result = np.zeros((5000, 600, 896, 1))
    for i in range(5):
        print("Processing {}".format(i + 1), end="\r")
        result[i, :, :, 0] = np.loadtxt("/home/retina/skw/work/leap/leap/models/video1-leap_cnn_epochs=10_04/result.txt")
    print("Result shape: ", result.shape)
    # pd.DataFrame(result).to_csv("/home/retina/skw/work/leap/leap/models/video1-leap_cnn_epochs=10_04/result.csv")
    Yi = confmap[0:20, :, :, 0]
    Y = Yi.reshape(Yi.shape[0], -1)
    print(Y.shape)
    Y = np.argmax(Y, axis=-1)
    gt_coord = np.unravel_index(Y.reshape(20, -1), Yi.shape)
    gt_coord = np.concatenate(gt_coord, axis=-1)
    print(gt_coord)
    print(gt_coord.shape)
    
    Yi = result[0:20, :, :, 0]
    Y = Yi.reshape(Yi.shape[0], -1)
    print(Y.shape)
    Y = np.argmax(Y, axis=-1)
    result_coord = np.unravel_index(Y.reshape(20, -1), Yi.shape)
    result_coord = np.concatenate(result_coord, axis=-1)
    print(result_coord)

    res = np.linalg.norm(gt_coord - result_coord, axis=-1, keepdims=True)
    res = res < 20
    res = res.astype(np.int32)
    res = np.mean(res)
    print("Accuracy: ", res)


label_dir = "../../leap-origin/data/label/"
label_path = [
    'left-clear-video1-back_foot_left.txt',
    'left-clear-video1-front_foot_left.txt',
    'left-clear-video2-back_foot_left.txt',
    'left-clear-video2-front_foot_left.txt',     
    'left-clear-video3-back_foot_left.txt',
    'left-clear-video3-front_foot_left.txt',
    'left-light-video4-back_foot_left.txt',
    'left-light-video4-front_foot_left.txt',
    'left-light-video5-back_foot_left.txt',
    'left-light-video5-front_foot_left.txt',
    'left-light-video6-back_foot_left.txt',
    'left-light-video6-front_foot_left.txt',
    'right-light-video7-back_foot_right.txt',
    'right-light-video7-front_foot_right.txt',
    'right-light-video8-back_foot_right.txt'
    'right-light-video8-front_foot_right.txt'
]

confmap_dir = '../data/confmap/'
video_dir = "../../leap-origin/data/h5video/video"
video_file = [video_dir + str(i) + '.h5' for i in range(10)]

if __name__ == "__main__":
    # Turn interactive plotting off
    # plt.ioff()

    # Wrapper for running from commandline
    # clize.run(train)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_video_path", type=str, required=True, help="The path of the training video with h5 format")
    parser.add_argument("--train_video_path_extra", type=str, default=None, required=False, help="Add another video path")   
    parser.add_argument("--label_path", type=str, required=True, help="The path of the label file with format: [frame_idx, x, y, w, h]")
    parser.add_argument("--label_path_extra", type=str, default=None, help="Add another label path")
    parser.add_argument("--base_output_path", type=str, default="models", help="The base output path to store the model and visualizaiton results")
    parser.add_argument("--test_video_path", type=str, default=None, help="The test video path")
    parser.add_argument("--test_video_path_extra", type=str, default=None, help="Add another test video path")
    parser.add_argument("--test_label_path", type=str, default=None, help="Testing data label path")
    parser.add_argument("--test_label_path_extra", type=str, default=None, help="Add another testing label path")
    parser.add_argument("--train_size", type=int, default=800, help="Training data size")
    parser.add_argument("--mode", type=str, default="same", help="same: train and test on the same dataset; diff: train and test on different datasets")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training")
    parser.add_argument("--test_idx", type=str, default=None, help="Test idx for test only")
    parser.add_argument("--model_path", type=str, default=None, help="Trained model path")
    parser.add_argument("--epoch", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    train_path = [args.train_video_path]
    if args.train_video_path_extra:
        train_path.append(args.train_video_path_extra)    

    label_path = [args.label_path]
    if args.label_path_extra:
        label_path.append(args.label_path_extra)
    
    test_path = list()
    if args.test_video_path:
        test_path.append(args.test_video_path)        
        if args.test_video_path_extra:
            test_path.append(args.test_video_path_extra)

    test_label_path = list()
    if args.test_label_path:
        test_label_path.append(args.test_label_path)
        if args.test_label_path_extra:
            test_label_path.append(args.test_label_path_extra)

    # train and test on the same video
    if args.mode == "same":
        train_test_same(train_path, label_path, batch_size=args.batch, epochs=args.epoch, train_size=args.train_size, base_output_path=args.base_output_path)
    elif args.mode == "diff":
        # train and test on different videos        
        train_test_diff(train_path, label_path, test_path, test_label_path, batch_size=args.batch, epochs=args.epoch, train_size=args.train_size, base_output_path=args.base_output_path)
    elif args.mode == "test":
        test(train_path, label_path, args.model_path, args.test_idx, batch_size=args.batch, base_output_path=args.base_output_path)
    else:
        print("--mode should be in [same, diff, test]")

    # test(train_path, label_path, base_output_path=args.base_output_path)
    # cal_acc(label_path)


    """label_dir = "../../leap-origin/data/label/"
    label_path = [
        'left-clear-video1-back_foot_left.txt',
        'left-clear-video1-front_foot_left.txt',
        'left-clear-video2-back_foot_left.txt',
        'left-clear-video2-front_foot_left.txt',     
        'left-clear-video3-back_foot_left.txt',
        'left-clear-video3-front_foot_left.txt',
        'left-light-video4-back_foot_left.txt',
        'left-light-video4-front_foot_left.txt',
        'left-light-video5-back_foot_left.txt',
        'left-light-video5-front_foot_left.txt',
        'left-light-video6-back_foot_left.txt',
        'left-light-video6-front_foot_left.txt',
        'right-light-video7-back_foot_right.txt',
        'right-light-video7-front_foot_right.txt',
        'right-light-video8-back_foot_right.txt'
        'right-light-video8-front_foot_right.txt',
        'right-light-video9-back_foot_right.txt',
        'right-light-video9-front_foot_right.txt',
        'left-light-video10-back_foot_left.txt',
        'left-light-video10-front_foot_left.txt',
        'left-light-video11-back_foot_left.txt',
        None,
        'right-light-video12-back_foot_right.txt',
        'right-light-video12-front_foot_right.txt',
        'right-light-video13-back_foot_right.txt',
        None
    ]

    confmap_dir = '../data/confmap/'
    video_dir = "../../leap-origin/data/h5video/video"
    video_file = [video_dir + str(i + 1) + '.h5' for i in range(13)]
    for i in range(1, 13):
        label = list()
        for t in range(2):
            index = i * 2 + t
            if label_path[index] is None:
                continue
            label.append(label_dir + label_path[index])
        print("Processing video ", i + 1)
        data_path = video_file[i]
        box = load_video(data_path, X_dset="box")
        confmap = load_label(label, *box.shape[:-1], channels=len(label))
        print("Conf map shape: ", confmap.shape)
        h5f = h5py.File('/media/retina/新加卷1/skw/work/tracking/leap/data/video' + str(i + 1), 'w')
        h5f.create_dataset('confmap', data=confmap)
        h5f.close()"""
