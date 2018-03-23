"""Define the model."""

import tensorflow as tf
import numpy as np

def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 1]

    # Define the number of channels of each convolution
    # For each block, we do: 5x5 conv -> batch norm -> relu -> 2x1 maxpool same padding
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum

    c=num_channels
    with tf.variable_scope('Left_{}'.format(1)):
        outL1 = tf.layers.conv2d(images, c, 3, padding='same')
        if params.use_batch_norm:
            outL1 = tf.layers.batch_normalization(outL1, momentum=bn_momentum, training=is_training)
        outL1 = tf.nn.relu(outL1)
    with tf.variable_scope('Left_{}'.format(2)):
        outL2 = tf.layers.max_pooling2d(outL1, 2, 2)
        outL2 = tf.layers.conv2d(outL2, c*2, 3, padding='same')
        if params.use_batch_norm:
            outL2 = tf.layers.batch_normalization(outL2, momentum=bn_momentum, training=is_training)
        outL2 = tf.nn.relu(outL2)
    with tf.variable_scope('Left_{}'.format(3)):
        outL3 = tf.layers.max_pooling2d(outL2, 2, 2)
        outL3 = tf.layers.conv2d(outL3, c*4, 3, padding='same')
        if params.use_batch_norm:
            outL3 = tf.layers.batch_normalization(outL3, momentum=bn_momentum, training=is_training)
        outL3 = tf.nn.relu(outL3)
    with tf.variable_scope('Left_{}'.format(4)):
        outL4 = tf.layers.max_pooling2d(outL3, 2, 2)
        outL4 = tf.layers.conv2d(outL4, c*8, 3, padding='same')
        if params.use_batch_norm:
            outL4 = tf.layers.batch_normalization(outL4, momentum=bn_momentum, training=is_training)
        outL4 = tf.nn.relu(outL4)
   
    with tf.variable_scope('Bottom'):
        out = tf.layers.max_pooling2d(outL4, 2, 2)
        out = tf.layers.conv2d(out, c*16, 3, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.conv2d(out,c*8,3,padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)

    with tf.variable_scope('Right_{}'.format(4)):
        out = tf.layers.conv2d_transpose(out, c*8, 3,strides=(2,2), padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.concat([outL4,out],-1)
        out = tf.layers.conv2d(out, c*4, 3, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('Right_{}'.format(3)):
        out = tf.layers.conv2d_transpose(out, c*4, 3,strides=(2,2), padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.concat([outL3,out],-1)
        out = tf.layers.conv2d(out, c*2, 3, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('Right_{}'.format(2)):
        out = tf.layers.conv2d_transpose(out, c*2, 3,strides=(2,2), padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.concat([outL2,out],-1)
        out = tf.layers.conv2d(out, c, 3, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('Right_{}'.format(1)):
        out = tf.layers.conv2d_transpose(out, c, 3,strides=(2,2), padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.concat([outL1,out],-1)
        out = tf.layers.conv2d(out, c, 3, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)

    out = tf.layers.conv2d(out, 1, 3, padding='same')
    out = tf.nn.relu(out)

    assert out.get_shape().as_list() == [None, params.image_size, params.image_size, 1]

    return out


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    images = inputs['images']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        predictions = build_model(is_training, inputs, params)

    # Define loss and similarity
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    size=params.image_size
    predictions_reshape=tf.reshape(predictions,[-1,size*size])
    predictions_reshape=tf.nn.l2_normalize(predictions_reshape, [1])

    labels_reshape=tf.reshape(labels,[-1,size*size])
    labels_reshape=tf.nn.l2_normalize(labels_reshape, [1])

    images_reshape=tf.reshape(images,[-1,size*size])
    images_reshape=tf.nn.l2_normalize(images_reshape, [1])

    orig_similarity=(tf.reduce_sum(tf.multiply(images_reshape,labels_reshape),axis=1))
    new_similarity=(tf.reduce_sum(tf.multiply(predictions_reshape,labels_reshape),axis=1))
    similarity_progress=tf.reduce_mean(new_similarity-orig_similarity)


    similarity=tf.reduce_mean(new_similarity)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'similarity': tf.metrics.mean(similarity),
            'loss': tf.metrics.mean(loss),
            'similarity_progress':tf.metrics.mean(similarity_progress)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('similarity', similarity)
    tf.summary.scalar('similarity_progress', similarity_progress)
#    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    similarity_arr=(tf.reduce_sum(tf.multiply(predictions_reshape,labels_reshape),axis=1))
    mask = tf.greater(0.5, similarity_arr)

    # Add a different summary to know how they were misclassified

    incorrect_train_image = tf.boolean_mask(inputs['images'], mask)
    tf.summary.image('incorrectly_train', incorrect_train_image)
    incorrect_predict_image = tf.boolean_mask(predictions, mask)
    tf.summary.image('incorrectly_predict',incorrect_predict_image)
    incorrect_image_label = tf.boolean_mask(labels, mask)
    tf.summary.image('incorrect_label',incorrect_image_label)



    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['similarity'] = similarity
    model_spec['similarity_progress'] = similarity_progress
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
