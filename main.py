#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import time
from datetime import timedelta
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    graph = tf.get_default_graph()
    # Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # The layers architectures:
    #  [vgg3]     [vgg4]     [vgg7]
    #    |          |          |
    #    |          |          \--->[conv2d:1x1]---\
    #    |          |                              |
    #    | [scale] [x]                             V
    #    |  0.01    |                      [conv2d transpose] #1
    #    |          |                              |
    #    |          |                              V
    #    |          \--->[conv2d:1x1]------------>[+] [add] #1
    #   [x] [scale]                                |
    #    |  0.0001                                 V
    #    |                                 [conv2d transpose] #2
    #    |                                         |
    #    |                                         V
    #    \-------------->[conv2d:1x1]------------>[+] [add] #2
    #                                              |
    #                                              V
    #                                      [conv2d transpose] (output layer)
    #
    
    # Constants
    PADDING_SAME = 'same'
    STDDEV = 1e-2
    SCALE_L2_REG = 1e-5
    
    # Scale vgg3 then fed it to a 1x1 convolutional network
    scale_vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001, name='scale_vgg_layer3_out')
    conv2d_1x1_layer_3 = tf.layers.conv2d(
        inputs=scale_vgg_layer3_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding=PADDING_SAME,
        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=SCALE_L2_REG),
        name='conv2d_1x1_layer_3')
    
    # Scale vgg4 then fed it to a 1x1 convolutional network
    scale_vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01, name='scale_vgg_layer4_out')
    conv2d_1x1_layer_4 = tf.layers.conv2d(
        inputs=scale_vgg_layer4_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding=PADDING_SAME,
        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=SCALE_L2_REG),
        name='conv2d_1x1_layer_4')
    
    # Fed vgg7 to a 1x1 convolutional network
    conv2d_1x1_layer_7 = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding=PADDING_SAME,
        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=SCALE_L2_REG),
        name='conv2d_1x1_layer_7')
    
    # Upsample the `conv2d_1x1_layer_7` layer
    conv2d_tr_1 = tf.layers.conv2d_transpose(
        inputs=conv2d_1x1_layer_7,
        filters=num_classes,
        kernel_size=4,
        strides=2,
        padding=PADDING_SAME,
        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=SCALE_L2_REG),
        name='conv2d_tr_1')
    
    # Add `conv2d_1x1_layer_4` to it
    add_layer_1 = tf.add(conv2d_tr_1, conv2d_1x1_layer_4, name='add_layer_1')
    
    # Upsample the `add_layer_1` layer
    conv2d_tr_2 = tf.layers.conv2d_transpose(
        inputs=add_layer_1,
        filters=num_classes,
        kernel_size=4,
        strides=2,
        padding=PADDING_SAME,
        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=SCALE_L2_REG),
        name='conv2d_tr_2')
    
    # Add `conv2d_1x1_layer_3` to it
    add_layer_2 = tf.add(conv2d_tr_2, conv2d_1x1_layer_3, name='add_layer_2')
    
    # Upsample the `add_layer_2` layer
    output_layer = tf.layers.conv2d_transpose(
        inputs=add_layer_2,
        filters=num_classes,
        kernel_size=16,
        strides=8,
        padding=PADDING_SAME,
        kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=SCALE_L2_REG),
        name='output_layer')
    
    return output_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    # logits = nn_last_layer
    # labels = correct_label
    
    # Loss function
    # If using exclusive labels (wherein one and only one class is true at a time), 
    # see `sparse_softmax_cross_entropy_with_logits`.
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
#     # We have to manually add all those regularization loss terms to
#     # your loss function, otherwise they are not doing anything.
#     regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # Scalar
#     cross_entropy_loss = cross_entropy_loss + regularization_loss

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Training operation
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # +---------------[ Hyperparameters ]---------------+
    keep_prob_value = 0.75
    learning_rate_value = 1e-4
    # +-------------------------------------------------+
    
    losses = []
    print(">> Start...")
    for epoch in range(epochs):
        loss = None
        s_time = time.time()
        for image, labels in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image: image,
                           correct_label: labels,
                           keep_prob: keep_prob_value,
                           learning_rate: learning_rate_value}
            )
            losses.append(loss)
        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch + 1, epochs, loss, str(timedelta(seconds=(time.time() - s_time)))))
        # print("[Epoch: {0}/{1} Loss: {2:4f}]".format(epoch + 1, epochs, loss))
    
    # Saving losses
    directory='./runs'
    print(">> Saving loss values to \"{0}\"".format(directory))
    helper.save_array_to_csv(losses, directory=directory, header='loss')
    print(">> End...\n")
tests.test_train_nn(train_nn)


def run():
    # +------------------[ Constants ]------------------+
    image_shape = (160, 576)
    num_classes = 2
    data_dir = '/data'
    data_dir1 = './data'
    runs_dir = './runs'
    # +-------------------------------------------------+
    
    tests.test_for_kitti_dataset(data_dir)
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # +---------------[ Hyperparameters ]---------------+
    epochs = 20
    batch_size = 8
    #epochs = 2
    #batch_size = 1
    # +-------------------------------------------------+

    with tf.Session() as sess:
        print("\n\n")
        print("+-------------------------------------------------+")
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        data_folder = os.path.join(data_dir, 'data_road/training')
        data_folder1 = os.path.join(data_dir1, 'data_road/training')
        helper.process_data(data_folder1, data_folder)
        get_batches_fn = helper.gen_batch_function(data_folder1, image_shape)

        # Define TF placeholders
        print(">> Define TF placeholders...")
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Build NN using load_vgg, layers, and optimize function
        print(">> Build NN using load_vgg, layers, and optimize function...")
        print("   + Loading VGG...")
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        print("   + Building network...")
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        print("   + Building optimizer and loss function...")
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        print(">> Global variable initializing...")
        sess.run(tf.global_variables_initializer())
        print(">> Training...")
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
