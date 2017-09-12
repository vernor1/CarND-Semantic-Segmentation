import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from functools import partial
from moviepy.editor import VideoFileClip
import project_tests as tests


# Local Constants
# ------------------------------------------------------------------------------

EPOCHS = 100
BATCH_SIZE = 50
KEEP_PROB = 0.7
LEARNING_RATE = 1e-3

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
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
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
    # Freeze layers of VGG16.
    #vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
    #vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    #vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    # VGG Layer 7, 1x1 convolution to predict scores for each of the classes.
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                num_classes,
                                1,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # 2x upsampling, building FCN-32.
    output = tf.layers.conv2d_transpose(conv_1x1,
                                        num_classes,
                                        4,
                                        strides=(2,2),
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # VGG Layer 4, 1x1 convolution to produce additional class predictions.
    conv_1x1 = tf.layers.conv2d(vgg_layer4_out,
                                num_classes,
                                1,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Fuse output of the FCN-32 predictions with the VGG Layer 4 predictions.
    output = tf.add(output, conv_1x1)
    # 2x upsampling, building FCN-16.
    output = tf.layers.conv2d_transpose(output,
                                        num_classes,
                                        4,
                                        strides=(2,2),
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # VGG Layer 3, 1x1 convolution to produce additional class predictions.
    conv_1x1 = tf.layers.conv2d(vgg_layer3_out,
                                num_classes,
                                1,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Fuse output of the FCN-16 predictions with the VGG Layer 3 predictions.
    output = tf.add(output, conv_1x1)
    # 8x upsampling, building FCN-8 and upsampling it to the original image size.
    output = tf.layers.conv2d_transpose(output,
                                        num_classes,
                                        16,
                                        strides=(8,8),
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFlow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def get_metrics(logits, correct_label, num_classes):
    """
    return: Tuple of (mean_iou, iou_op)
    """
    labels = tf.argmax(tf.reshape(correct_label, (-1, num_classes)), 1)
    predictions = tf.argmax(logits, 1)
    mean_iou, mean_op = tf.metrics.mean_iou(labels, predictions, num_classes)
    _, acc_op = tf.metrics.accuracy(labels, predictions)
    return mean_iou, mean_op, acc_op


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou_op=None, mean_iou=None, acc_op=None):
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
    for epoch in range(epochs):
      # Assume that the only local variable in use is the confusion matrix
      # updated by tf.metrics.mean_iou(). The matrix needs to be reset for
      # every epoch.
      sess.run(tf.local_variables_initializer())
      for images, labels in get_batches_fn(batch_size):
        feed_dict = {input_image: images, correct_label: labels, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE}
        _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
        if iou_op != None:
          _, miou, acc = sess.run([iou_op, mean_iou, acc_op], feed_dict=feed_dict)
      if iou_op == None:
        print("Epoch %d, loss %f" % (epoch + 1, loss))
      else:
        print("Epoch %d, loss %f, mIOU %f, accuracy %f" % (epoch + 1, loss, miou, acc))
    pass
tests.test_train_nn(train_nn)

def process_image(sess, image_input, logits, keep_prob, image_shape, in_img):
    """ Processes an RGB image for detecting vehicles.
        The information is added to the original image in overlay.

    param: img: Image to process
    returns: Processed RGB image
    """
    in_img = in_img[300:-65,1:-1,:]
    in_img = scipy.misc.imresize(in_img, (image_shape[0] * 2, image_shape[1] * 2))
    img = scipy.misc.imresize(in_img, image_shape)
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_input: [img]})
    im_softmax = im_softmax[0][:, 1].reshape(img.shape[0], img.shape[1])
    segmentation = (im_softmax > 0.5).reshape(img.shape[0], img.shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.imresize(mask, (in_img.shape[0], in_img.shape[1]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(in_img)
    street_im.paste(mask, box=None, mask=mask)
    out_img = np.array(street_im)
    scipy.misc.imsave("test.png", out_img)
    return out_img

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        correct_label = tf.placeholder(tf.float32, (None, None, None, num_classes))
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        num_classes)

        mean_iou, iou_op, acc_op = get_metrics(logits,
                                               correct_label,
                                               num_classes)

        sess.run(tf.global_variables_initializer())

        # Load model.
        tf.train.Saver().restore(sess, "./checkpoints/model.ckpt")

        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op,
                 cross_entropy_loss, image_input, correct_label, keep_prob,
                 learning_rate, iou_op, mean_iou, acc_op)

        # Save model.
        tf.train.Saver().save(sess, "./checkpoints/model.ckpt")

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video
        in_clip = VideoFileClip("project_video.mp4")
        parametrized_process_image = partial(process_image, sess, image_input, logits, keep_prob, image_shape)
        out_clip = in_clip.fl_image(parametrized_process_image)
        out_clip.write_videofile("out_video.mp4", audio=False)


if __name__ == '__main__':
    run()
