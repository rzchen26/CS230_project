"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np

def _parse_function(filename, size):
    """Obtain the image X and Y from the filename (for both training and validation).

    The following operations are applied:
        - numpy.load from .npz file 
    """
    keys_to_features = {'X':tf.FixedLenFeature((512,512,1), tf.float32),
                        'y':tf.FixedLenFeature((512,512,1), tf.float32)}
    parsed_features = tf.parse_single_example(filename, keys_to_features)
    
    image=parsed_features['X']

    label=parsed_features['y']

    resized_image = tf.image.resize_images(image, [size, size])
    resized_label = tf.image.resize_images(label, [size, size])

    return resized_image, resized_label


def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
    """
    if use_random_flip:
        my_seed = np.random.randint(0, 2 ** 31 - 1)
        image = tf.image.random_flip_left_right(image,seed=my_seed)
        label = tf.image.random_flip_left_right(label,seed=my_seed)

    return image, label


def input_fn(is_training, filenames, params):
    """Input function for  dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the pairs of image and label
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f: _parse_function(f,  params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
         dataset = (tf.data.TFRecordDataset(tf.constant(filenames))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
         dataset = (tf.data.TFRecordDataset(tf.constant(filenames))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
