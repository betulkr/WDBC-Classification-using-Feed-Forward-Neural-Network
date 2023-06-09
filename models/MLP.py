# Copyright 2021 Fatma Betul Kara Ardac
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Implementation of the Multilayer Perceptron using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Fatma Betul Kara Ardac"

import numpy as np
import os
import sys
import time
import tensorflow as tf


class MLP:
    """Implementation of the Multilayer Perceptron using TensorFlow"""

    def __init__(self, alpha, batch_size, node_size, num_classes, num_features):
        """Initialize the MLP model

        Parameter
        ---------
        alpha : float
          The learning rate to be used by the neural network.
        batch_size : int
          The number of batches to use for training/validation/testing.
        node_size : int
          The number of neurons in the neural network.
        num_classes : int
          The number of classes in a dataset.
        num_features : int
          The number of features in a dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.node_size = node_size
        self.num_classes = num_classes
        self.num_features = num_features

        def __graph__():
            """Build the inference graph"""

            with tf.name_scope("input"):
                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.num_features], name="x_input"
                )

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name="y_input")

                # [BATCH_SIZE, NUM_CLASSES]
                y_onehot = tf.one_hot(
                    indices=y_input,
                    depth=self.num_classes,
                    on_value=1,
                    off_value=0,
                    name="y_onehot",
                )

            learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

            first_hidden_layer = {
                "weights": self.weight_variable(
                    "h1_w_layer", [self.num_features, self.node_size[0]]
                ),
                "biases": self.bias_variable("h1_b_layer", [self.node_size[0]]),
            }

            second_hidden_layer = {
                "weights": self.weight_variable(
                    "h2_w_layer", [self.node_size[0], self.node_size[1]]
                ),
                "biases": self.bias_variable("h2_b_layer", [self.node_size[1]]),
            }

            third_hidden_layer = {
                "weights": self.weight_variable(
                    "h3_w_layer", [self.node_size[1], self.node_size[2]]
                ),
                "biases": self.bias_variable("h3_b_layer", [self.node_size[2]]),
            }

            output_layer = {
                "weights": self.weight_variable(
                    "output_w_layer", [self.node_size[2], self.num_classes]
                ),
                "biases": self.bias_variable("output_b_layer", [self.num_classes]),
            }

            first_layer = (
                tf.matmul(x_input, first_hidden_layer["weights"])
                + first_hidden_layer["biases"]
            )
            first_layer = tf.nn.relu(first_layer)

            second_layer = (
                tf.matmul(first_layer, second_hidden_layer["weights"])
                + second_hidden_layer["biases"]
            )
            second_layer = tf.nn.relu(second_layer)

            third_layer = (
                tf.matmul(second_layer, third_hidden_layer["weights"])
                + third_hidden_layer["biases"]
            )
            third_layer = tf.nn.relu(third_layer)

            output_layer = (
                tf.matmul(third_layer, output_layer["weights"]) + output_layer["biases"]
            )
            tf.summary.histogram("pre-activations", output_layer)

            with tf.name_scope("loss"):
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=output_layer, labels=y_onehot
                    )
                )
            tf.summary.scalar("loss", loss)

            optimizer_op = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            ).minimize(loss)

            with tf.name_scope("accuracy"):
                predicted_class = tf.nn.softmax(output_layer)
                with tf.name_scope("correct_prediction"):
                    correct_prediction = tf.equal(
                        tf.argmax(predicted_class, 1), tf.argmax(y_onehot, 1)
                    )
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar("accuracy", accuracy)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.y_onehot = y_onehot
            self.learning_rate = learning_rate
            self.loss = loss
            self.optimizer_op = optimizer_op
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write("\n<log> Building Graph...")
        __graph__()
        sys.stdout.write("</log>\n")

    def train(
        self,
        num_epochs,
        log_path,
        train_data,
        train_size,
        test_data,
        test_size,
        result_path,
    ):
        """Trains the MLP model

        Parameter
        ---------
        num_epochs : int
          The number of passes over the entire dataset.
        log_path : str
          The path where to save the TensorBoard logs.
        train_data : numpy.ndarray
          The NumPy array to be used as training dataset.
        train_size : int
          The size of the `train_data`.
        test_data : numpy.ndarray
          The NumPy array to be used as testing dataset.
        test_size : int
          The size of the `test_data`.
        """

        # initialize the variables
        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )

        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(
            log_path + timestamp + "-training", graph=tf.get_default_graph()
        )
        test_writer = tf.summary.FileWriter(
            log_path + timestamp + "-test", graph=tf.get_default_graph()
        )

        with tf.Session() as sess:
            sess.run(init_op)

            try:
                for step in range(num_epochs * train_size // self.batch_size):
                    offset = (step * self.batch_size) % train_size
                    train_data_batch = train_data[0][
                        offset : (offset + self.batch_size)
                    ]
                    train_label_batch = train_data[1][
                        offset : (offset + self.batch_size)
                    ]

                    feed_dict = {
                        self.x_input: train_data_batch,
                        self.y_input: train_label_batch,
                        self.learning_rate: self.alpha,
                    }

                    train_summary, _, step_loss = sess.run(
                        [self.merged, self.optimizer_op, self.loss], feed_dict=feed_dict
                    )

                    if step % 100 == 0 and step > 0:
                        train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
                        print(
                            "step [{}] train -- loss : {}, accuracy : {}".format(
                                step, step_loss, train_accuracy
                            )
                        )
                        train_writer.add_summary(train_summary, global_step=step)

            except KeyboardInterrupt:
                print("KeyboardInterrupt at step {}".format(step))
                os._exit(1)
            finally:
                print("EOF -- Training done at step {}".format(step))

                for step in range(num_epochs * test_size // self.batch_size):
                    offset = (step * self.batch_size) % test_size
                    test_data_batch = test_data[0][offset : (offset + self.batch_size)]
                    test_label_batch = test_data[1][offset : (offset + self.batch_size)]

                    feed_dict = {
                        self.x_input: test_data_batch,
                        self.y_input: test_label_batch,
                    }

                    (
                        test_summary,
                        test_accuracy,
                        test_loss,
                        predictions,
                        actual,
                    ) = sess.run(
                        [
                            self.merged,
                            self.accuracy,
                            self.loss,
                            self.predicted_class,
                            self.y_onehot,
                        ],
                        feed_dict=feed_dict,
                    )

                    if step % 100 == 0 and step > 0:
                        print(
                            "step [{}] test -- loss : {}, accuracy : {}".format(
                                step, test_loss, test_accuracy
                            )
                        )
                        test_writer.add_summary(test_summary, step)

                    self.save_labels(
                        predictions=predictions,
                        actual=actual,
                        result_path=result_path,
                        phase="testing",
                        step=step,
                    )

                print("EOF -- Testing done at step {}".format(step))

    @staticmethod
    def weight_variable(name, shape):
        """Initialize weight variable

        Parameter
        ---------
        shape : list
          The shape of the initialized value.

        Returns
        -------
        The created `tf.get_variable` for weights.
        """
        initial_value = tf.random_normal(shape=shape, stddev=0.01)
        return tf.get_variable(name=name, initializer=initial_value)

    @staticmethod
    def bias_variable(name, shape):
        """Initialize bias variable

        Parameter
        ---------
        shape : list
          The shape of the initialized value.

        Returns
        -------
        The created `tf.get_variable` for biases.
        """
        initial_value = tf.constant([0.1], shape=shape)
        return tf.get_variable(name=name, initializer=initial_value)

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)

    @staticmethod
    def save_labels(predictions, actual, result_path, phase, step):
        """Saves the actual and predicted labels to a NPY file

        Parameter
        ---------
        predictions : numpy.ndarray
          The NumPy array containing the predicted labels.
        actual : numpy.ndarray
          The NumPy array containing the actual labels.
        result_path : str
          The path where to save the concatenated actual and predicted labels.
        step : int
          The time step for the NumPy arrays.
        phase : str
          The phase for which the predictions is, i.e. training/validation/testing.
        """

        if not os.path.exists(path=result_path):
            os.mkdir(result_path)

        # Concatenate the predicted and actual labels
        labels = np.concatenate((predictions, actual), axis=1)

        # save every labels array to NPY file
        np.save(
            file=os.path.join(result_path, "{}-mlp-{}.npy".format(phase, step)),
            arr=labels,
        )
