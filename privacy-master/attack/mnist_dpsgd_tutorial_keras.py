# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer , DPAdamGaussianOptimizer, DPAdagradGaussianOptimizer

GradientDescentOptimizer = tf.train.GradientDescentOptimizer
AdamOptimizer = tf.train.AdamOptimizer
AdagradOptimizer = tf.train.AdagradOptimizer
tf.compat.v1.disable_eager_execution()

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('dataset', "MNIST", 'Dataset used')
flags.DEFINE_integer('data_slice', 60000, 'dataset size')
FLAGS = flags.FLAGS


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
  test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  #train_data = train_data[:FLAGS.data_slice,:]
  #train_labels = train_labels[:FLAGS.data_slice, :]
  #test_data = test_data[:FLAGS.data_slice, :]
  #test_labels = test_labels[:FLAGS.data_slice, :]

  return train_data, train_labels, test_data, test_labels


def main():
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  #train_data, train_labels, test_data, test_labels = load_mnist()

  # Define a sequential Keras model
  model = tf.keras.Sequential()
  if FLAGS.dataset == "cifar10":
      model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                       input_shape=(32,32,3)))
      model.add(tf.keras.layers.Activation('relu'))
      model.add(tf.keras.layers.Conv2D(32, (3, 3)))
      model.add(tf.keras.layers.Activation('relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Dropout(0.25))

      model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
      model.add(tf.keras.layers.Activation('relu'))
      model.add(tf.keras.layers.Conv2D(64, (3, 3)))
      model.add(tf.keras.layers.Activation('relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
      model.add(tf.keras.layers.Dropout(0.25))

      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(512))
      model.add(tf.keras.layers.Activation('relu'))
      model.add(tf.keras.layers.Dropout(0.5))
      model.add(tf.keras.layers.Dense(10))
      model.add(tf.keras.layers.Activation('softmax'))
  elif FLAGS.dataset =="MNIST":
      model.add( tf.keras.layers.Conv2D(16, 8,
                                 # strides=2,
                                 padding='same',
                                 activation='relu',
                                 input_shape=(28, 28, 1)))
      model.add( tf.keras.layers.MaxPool2D(2, 1))
      model.add(tf.keras.layers.Conv2D(32, 4,
                                 # strides=2,
                                 padding='valid',
                                 activation='relu'))
      model.add(tf.keras.layers.MaxPool2D(2, 1))
      model.add(tf.keras.layers.Flatten())
      model.add( tf.keras.layers.Dense(32, activation='relu'))
      model.add(tf.keras.layers.Dense(10, activation='softmax'))



  if FLAGS.dpsgd:
    optimizer = DPAdamGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
  else:
    optimizer = AdamOptimizer(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # Train model with Keras
  return model
if __name__ == '__main__':
  app.run(main)
