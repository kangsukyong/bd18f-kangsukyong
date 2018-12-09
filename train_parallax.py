# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
python train_parallax.py --input_file_pattern={processed_data_path}/train-?????-of-00256 --inception_checkpoint_file={inception_v3_file} --train_inception=false --number_of_steps=200

Example command for examining the checkpoint file:
python <PARALLAX_HOME>/tensorflow/tensorflow/python/tools/inspect_checkpoint.py --file_name=tf_ckpt/model.ckpt-0 --tensor_name=InceptionV3/Conv2d_a1_3x3_weights
"""

"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import parallax

from im2txt import configuration
from im2txt import show_and_tell_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "", "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("inception_checkpoint_file", "", "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_boolean("train_inception", False, "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10, "Frequency at which loss and global step are logged.")

tf.flags.DEFINE_string("resource_info_file", 
                       os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "resource_info")),
                       "Filename containing cluster information")

tf.flags.DEFINE_boolean("sync", True, '')

tf.logging.set_verbosity(tf.logging.INFO)

assert FLAGS.input_file_pattern, "--input_file_pattern is required"
assert FLAGS.train_dir, "--train_dir is required"

model_config = configuration.ModelConfig()
model_config.input_file_pattern = FLAGS.input_file_pattern
model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file

training_config = configuration.TrainingConfig()

g = tf.Graph()
with g.as_default():
    # build the model
    model = show_and_tell_model.ShowAndTellModel(model_config, mode='train', train_inception=FLAGS.train_inception)
    model.build()

    # set up the learning rate
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
        learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
        learning_rate = tf.constant(training_config.initial_learning_rate)

        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = training_config.num_examples_per_epoch / model_config.batch_size
            decay_steps = int(num_batches_per_epoch * training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(learning_rate, 
                                                  global_step, 
                                                  decay_steps=decay_steps, 
                                                  decay_rate=training_config.learning_rate_decay_factor, 
                                                  staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn
    
    # set up the training ops
    train_op = tf.contrib.layers.optimize_loss(loss=model.total_loss,
                                               global_step=model.global_step,
                                               learning_rate=learning_rate,
                                               optimizer=training_config.optimizer,
                                               clip_gradients=training_config.clip_gradients,
                                               learning_rate_decay_fn=learning_rate_decay_fn)

parallax_config = parallax.Config()
parallax_config.ckpt_config = parallax.CheckPointConfig(ckpt_dir='parallax_ckpt', save_ckpt_steps=1)

sess, num_workers, worker_id, num_replicas_per_worker = parallax.parallel_run(g,
                                                                              FLAGS.resource_info_file,
                                                                              sync=FLAGS.sync,
                                                                              parallax_config=parallax_config)

start = time.time()
for i in range(FLAGS.number_of_steps):
    if not sess.should_stop():
        _, loss_ = sess.run([train_op, model.total_loss])
        
        if i % FLAGS.log_every_n_steps  == 0:
            end = time.time()
            throughput = float(FLAGS.log_every_n_steps) / float(end - start)
            print("step: %d, throuphput: %f steps/sec" % (i, throughput))

            start = time.time()
