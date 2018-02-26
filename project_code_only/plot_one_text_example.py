"""Evaluate the model"""
from astropy.io import fits
import argparse
import logging
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from astropy.io import fits
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "plot")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.tfrecord')]


    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
#    evaluate(model_spec, args.model_dir, params, args.restore_from)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(model_spec['variable_init_op'])

        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
	prediction=sess.run(model_spec['predictions']).reshape([params.image_size,-1])
	

    with tf.Session() as sess:
        sess.run(model_spec['variable_init_op'])

        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
	orig_image=sess.run(model_spec['images']).reshape([params.image_size,-1])

    with tf.Session() as sess:
        sess.run(model_spec['variable_init_op'])

        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
        orig_labels=sess.run(model_spec['labels']).reshape([params.image_size,-1])
    
    vmin=0
    vmax=0.7
    filename_orig_image=os.path.join(arg.model_dir,  'AIA_example.jpg')
    filename_orig_label=os.path.join(arg.model_dir,   'Mag_example.jpg')
    filename_predictions=os.path.join(arg.model_dir,  'prediction_example.jpg')
    plt.imsave(filename_orig_image,orig_image,cmap='Blues',vmin=vmin,vmax=vmax)
    plt.imsave(filename_orig_label,orig_labels,cmap='Blues',vmin=vmin,vmax=vmax)
    plt.imsave(filename_predictions,predictions,cmap='Blues',vmin=vmin,vmax=vmax)
    fits.writeto(os.path.join(arg.model_dir, 'X_eg.fits'), orig_image )
    fits.writeto(os.path.join(arg.model_dir, 'Y_eg.fits'), orig_labels )
    fits.writeto(os.path.join(arg.model_dir, 'Y_prdct_eg.fits'), predictions)
    np.savez(os.path.join(arg.model_dir, 'X_Y_Yprdct_eg.npz'),orig_image,orig_labels,predictions)
