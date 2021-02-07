"""Math utils functions."""
from __future__ import print_function
import tensorflow as tf


def cosh(x, clamp=15):
    return tf.cosh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))


def sinh(x, clamp=15):
    return tf.sinh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))


def tanh(x, clamp=15):
    return tf.tanh(tf.clip_by_value(x, clip_value_min=-clamp, clip_value_max=clamp))


def arcosh(x):
    max_value = tf.reduce_max(x)
    if x.dtype == tf.float32:
        max_value = tf.cond(max_value < 1.0+1e-7, lambda: 1.0+1e-7, lambda: max_value)
        result = tf.acosh(tf.clip_by_value(x, clip_value_min=1.0+1e-7, clip_value_max=max_value)) #1e38
    elif x.dtype == tf.float64:
        max_value = tf.cond(max_value < 1.0+1e-16, lambda: 1.0+1e-16, lambda: max_value)
        result = tf.acosh(tf.clip_by_value(x, clip_value_min=1.0+1e-16, clip_value_max=max_value))
    else:
        raise ValueError('invalid dtype!')
    return result


def arsinh(x):
    result = tf.asinh(x)
    return result


def artanh(x):
    if x.dtype == tf.float32:
        result = tf.atanh(tf.clip_by_value(x, clip_value_min=tf.constant([-1], dtype=tf.float32)+1e-7, clip_value_max=tf.constant([1],dtype=tf.float32)-1e-7))
    elif x.dtype == tf.float64:
        result = tf.atanh(tf.clip_by_value(x, clip_value_min=tf.constant([-1], dtype=tf.float64)+1e-16, clip_value_max=tf.constant([1],dtype=tf.float64)-1e-16))
    else:
        raise ValueError('invalid dtype!')
    return result
