#!/usr/bin/env python3
import tensorflow as tf


def self_regularization_loss(y_true,y_pred):
    return tf.multiply(0.0002,tf.reduce_sum(tf.abs(y_pred-y_true)))

def local_adversarial_loss(y_true,y_pred):
    truth = tf.reshape(y_true,(-1,2))
    predicted = tf.reshape(y_pred,(-1,2))

    computed_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth,logits=predicted)
    output = tf.reduce_mean(computed_loss)
    return output