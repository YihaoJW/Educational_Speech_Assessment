import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from pathlib import Path
from tensorflow import Tensor
from tqdm.notebook import tqdm
from typing import Callable, List, Tuple, Union, Optional, Dict, Any, Sequence, Iterable, TypeVar
from DataPipe import DataPipeFactory
from util_function import inform_pooling, GatedXVector, PositionEncoding1D


# A function that generate a residual Block using separable convolution
def residual_block(x, channels, filter_size):
    x_input = x
    # Residual block start Normalize, Activate, and Convolution
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv1D(channels, filter_size, padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv1D(channels, filter_size, padding='same')(x)

    return tf.keras.layers.Add()([x_input, x])


# Build a Convolutional with to adjust the input to the residual block and apply several residual block
def residual_block_stack(x, channels, filter_size, stack_size):
    x = tf.keras.layers.Conv1D(channels, filter_size, padding='same')(x)
    for i in range(stack_size):
        x = residual_block(x, channels, filter_size)
    return x


# Define a Concatenate layer that will cut the input to the same length
class CutConcatenate(tf.keras.layers.Concatenate):
    def call(self, inputs):
        inter = [tf.slice(x, [0, 0, 0], [-1, tf.shape(inputs[0])[1], -1]) for x in inputs]
        # Set shape of inter to the shape of inputs if input is Tensor Placeholder
        shapes = [x.shape for x in inputs]
        for y, x in zip(inter, inputs):
            d = [sx if sx is not None and sy is None else sy for sx, sy in zip(x.shape, y.shape)]
            y.set_shape(d)

        return super().call(inter)


# Function build a U-Net
def build_unet(x, output_shape, channels_list, filter_size, stack_size):
    # Build the encoder
    encoder = []
    decoder = []
    for i in range(len(channels_list)):
        x = residual_block_stack(x, channels_list[i], filter_size, stack_size)
        encoder.append(x)
        if i < len(channels_list) - 1:
            x = tf.keras.layers.AvgPool1D(2)(x)
    # Build the decoder
    for i in range(len(channels_list) - 2, -1, -1):
        # Stride 2 convolution to upsample
        x = tf.keras.layers.Conv1DTranspose(channels_list[i], 4, strides=2, padding='valid')(x)
        x = CutConcatenate(axis=-1)([encoder[i], x])
        x = residual_block_stack(x, channels_list[i], filter_size, stack_size)
        decoder.append(x)
    # Build the output
    x = tf.keras.layers.Conv1D(output_shape, 1, padding='same')(x)
    return x, decoder


# Build a Network
def build_network(input_shape, output_shape, channels_list, filter_size, stack_size):
    x = tf.keras.Input(input_shape)
    y, maps = build_unet(x, output_shape, channels_list, filter_size, stack_size)
    return tf.keras.Model(x, y)


# Define a residual block that use fully connected layer
def residual_block_fc(x, channels):
    x_input = x
    # Residual block start Normalize, Activate, and Convolution
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(channels)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(channels)(x)

    return tf.keras.layers.Add()([x_input, x])


# %%
# Build a stack for fully connected layer
def residual_block_stack_fc(x, channels, stack_size):
    x = tf.keras.layers.Dense(channels)(x)
    for i in range(stack_size):
        x = residual_block_fc(x, channels)
    return x


# %%
# Create a dense residual network
def build_dense_network(input_shape, output_shape, channels_list, filter_size, stack_size):
    x = tf.keras.Input(input_shape)
    y = residual_block_stack(x, channels_list[0], filter_size, stack_size)
    for i in range(1, len(channels_list)):
        y = tf.keras.layers.Concatenate(axis=-1)([y, x])
        y = residual_block_stack(y, channels_list[i], filter_size, stack_size)
    y = tf.keras.layers.Conv1D(output_shape, 1, padding='same')(y)
    return tf.keras.Model(x, y)


# Define a Keras layers that perform information pooling
class InformPooling(tf.keras.layers.Layer):
    def __init__(self, num_maps, ratios_list, **kwargs):
        super().__init__(**kwargs)
        self.num_maps_shape = None
        self.num_maps = num_maps
        self.ratios_list = ratios_list

    def build(self, input_shape):
        # Input is a bunch of tensor, calcuate the total number of feature maps
        self.num_maps_shape = sum([x[-1] for x in input_shape])
        super().build(input_shape)

    @staticmethod
    @tf.function
    def inform_pooling(value, start, duration, ratio, eps=0.001):
        batch = tf.shape(value)[0]
        end = start + duration
        start = tf.math.floor(start * ratio)
        end = tf.math.ceil((end + eps) * ratio)
        period = tf.cast(tf.stack([start, end], axis=-1), tf.int32)
        # tf.debugging.assert_less(period[..., 0], period[..., 1])
        ret_b = tf.TensorArray(tf.float32, batch, infer_shape=False)
        ret_count = tf.TensorArray(tf.int64, batch)
        for batch_index in tf.range(batch):
            value_l = value[batch_index]
            val_ind_max = tf.shape(value_l)[0]
            period_l = period[batch_index]
            period_l_p = tf.math.minimum(period_l, val_ind_max - 1)
            ret_count = ret_count.write(batch_index, tf.cast(tf.shape(period_l)[0], tf.int64))
            indexes = tf.ragged.range(period_l_p[..., 0], period_l_p[..., 1])
            value_indices = tf.gather(value_l, indexes)
            pooled = tf.reduce_mean(value_indices, axis=1)
            ret_b = ret_b.write(batch_index, pooled)
        row_length = ret_count.stack()
        ret = ret_b.concat()
        return ret, row_length

    @tf.function
    def call(self, value_list, start, duration):
        # Iterate over both value_list and ratio_list
        pooled_value = [self.inform_pooling(value, start, duration, ratio) for (value, ratio) in
                        zip(value_list, self.ratios_list)]
        ret = tf.concat([val for val, _ in pooled_value], axis=-1)
        # Remove nan value to zero
        ret = tf.where(tf.math.is_nan(ret), 0., ret)
        # Stupid way to shrink dynamic shape to static shape
        ret.set_shape([None, self.num_maps_shape])
        ret = tf.RaggedTensor.from_row_lengths(ret, pooled_value[0][1])
        return ret


class ASR_Network(tf.keras.Model):
    def __init__(self, base_feature, dense_feature, word_prediction, base_ratio, batch_num, **kwargs):
        super().__init__()
        self.base_network = self.create_base_network(**base_feature)
        self.deep_feature = self.build_dense_network(**dense_feature)
        self.word_prediction = self.build_dense_network(**word_prediction)
        pooling_ratios = [base_ratio / 2 ** i for i in range(len(base_feature['channels_list']))]
        self.pooling = InformPooling(len(pooling_ratios), pooling_ratios)
        # define metrics
        self.loss_metrics = tf.keras.metrics.Mean(name='train_loss')
        self.word_loss_metric = tf.keras.metrics.Mean(name='train_word_loss')
        self.word_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_word_acc')
        self.deep_loss_metric = tf.keras.metrics.Mean(name='train_deep_loss')
        # define loss
        self.category_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.batch_counts = tf.Variable(batch_num, dtype=tf.int64, trainable=False)

    @staticmethod
    def create_base_network(input_shape, feature_depth, channels_list, filter_size, stack_size):
        x = tf.keras.Input(input_shape)
        y, maps = build_unet(x, feature_depth, channels_list, filter_size, stack_size)
        return tf.keras.Model(x, [y, maps])

    @staticmethod
    def build_dense_network(input_shape, output_shape, channels_list, stack_size):
        x = tf.keras.Input(input_shape)
        y = x
        for i in range(len(channels_list)):
            y = residual_block_stack_fc(y, channels_list[i], stack_size)
        y = tf.keras.layers.Dense(output_shape)(y)
        return tf.keras.Model(x, y)

    @staticmethod
    @tf.function
    def compute_similarity(value_a, value_b, ref_a, ref_b, margin=0.1, eps=0.01):
        # if ref_a equal ref_b then we consider it should be similar else it should be different,
        # margin prevent it been push to far away
        # compute the norm for ragged tensor
        norm_of_a = tf.sqrt(tf.reduce_sum(tf.square(value_a), axis=-1, keepdims=True))
        norm_of_b = tf.sqrt(tf.reduce_sum(tf.square(value_b), axis=-1, keepdims=True))
        norm_a = value_a / (norm_of_a + eps)
        norm_b = value_b / (norm_of_b + eps)
        # compute cosine similarity for each sample in batch
        # get batch size
        batch_size = tf.shape(norm_a)[0]
        loss_array = tf.TensorArray(tf.float32, batch_size, infer_shape=False)
        for idx in tf.range(batch_size):
            va = norm_a[idx]
            vb = norm_b[idx]
            ra = ref_a[idx]
            rb = ref_b[idx]
            similarity_matrix = tf.matmul(va, vb, transpose_b=True)
            # compute the mask for the positive samples
            mask = tf.cast(tf.equal(ra, rb), tf.float32)
            # compute the mask for the negative samples
            mask_neg = tf.cast(tf.not_equal(ra, rb), tf.float32)
            # compute the number of positive and negative samples
            num_pos = tf.cast(tf.reduce_sum(mask), tf.float32)
            num_neg = tf.cast(tf.reduce_sum(mask_neg), tf.float32)
            # compute the average similarity for the positive samples
            # avoid 0
            num_pos = tf.maximum(num_pos, 1.)
            num_neg = tf.maximum(num_neg, 1.)
            avg_sim_pos = tf.reduce_sum(tf.multiply(similarity_matrix, mask)) / num_pos
            # compute the average similarity for the negative samples
            avg_sim_neg = tf.reduce_sum(tf.multiply(similarity_matrix, mask_neg)) / num_neg
            # compute the max similarity for the positive samples
            max_sim_pos = tf.reduce_max(tf.multiply(similarity_matrix, mask))
            # compute the min similarity for the negative samples
            min_sim_neg = tf.reduce_min(tf.multiply(similarity_matrix, mask_neg))
            # compute the average loss with margin
            loss_avg = tf.maximum(0., margin + avg_sim_pos - avg_sim_neg)
            # compute min_max loss with margin
            loss_min_max = tf.maximum(0., margin + max_sim_pos - min_sim_neg)
            # total loss
            loss = loss_avg + loss_min_max
            loss_array = loss_array.write(idx, loss)
        total_loss = tf.reduce_sum(loss_array.stack())
        return total_loss

    #    @tf.function
    def call(self, inputs, training=False, mask=None):
        audio, (start, duration) = inputs
        # compute the base network
        base_output, maps = self.base_network(audio, training=training)
        # combine base output and maps
        total_maps = [base_output] + maps
        # pooling the total maps
        pooled_maps = self.pooling(total_maps, start, duration)
        # compute the deep feature
        deep_feature = tf.ragged.map_flat_values(lambda x: self.deep_feature(x, training=training, mask=mask), pooled_maps)
        # compute the word prediction
        word_prediction = tf.ragged.map_flat_values(lambda x: self.word_prediction(x, training=training, mask=mask), deep_feature)
        return word_prediction, deep_feature

    # compute a input pair
    def compute_pair(self, inputs, training=False, mask=None):
        student, reference = inputs
        # compute both student and reference
        student_output, student_deep_feature = self(student, training=training, mask=mask)
        reference_output, reference_deep_feature = self(reference, training=training, mask=mask)
        return (student_output, student_deep_feature), (reference_output, reference_deep_feature)

    # get loss for an input pair
    def compute_loss_pair(self, inputs, word_reference):
        (student_output, student_deep_feature), (reference_output, reference_deep_feature) = inputs
        # compute the loss for word prediction
        word_loss_student = self.category_loss(word_reference.flat_values, student_output.flat_values)
        word_loss_reference = self.category_loss(word_reference.flat_values, reference_output.flat_values)
        avg_word_loss = tf.reduce_sum((word_loss_student + word_loss_reference) / 2.) / tf.cast(self.batch_counts, tf.float32)
        # compute the loss for deep feature
        deep_loss = self.compute_similarity(student_deep_feature, reference_deep_feature, word_reference,
                                            word_reference) / tf.cast(self.batch_counts, tf.float32)
        return avg_word_loss, deep_loss

    def train_step(self, data):
        # unpack the data, input has two pair of audio and y has one word reference
        x, word_reference = data
        # compute the loss for each pair
        with tf.GradientTape() as tape:
            # compute the loss for each pair
            pair_data = self.compute_pair(x, training=True)
            avg_word_loss, deep_loss = self.compute_loss_pair(pair_data, word_reference)
            # compute the total loss
            total_loss = avg_word_loss + deep_loss
        # compute the gradient
        gradients = tape.gradient(total_loss, self.trainable_variables)
        # apply the gradient
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # update the metrics
        (student_output, _), (reference_output, _) = pair_data
        self.loss_metrics.update_state(total_loss)
        self.word_loss_metric.update_state(avg_word_loss)
        self.deep_loss_metric.update_state(deep_loss)
        self.word_acc_metric.update_state(word_reference.flat_values, student_output.flat_values)
        self.word_acc_metric.update_state(word_reference.flat_values, reference_output.flat_values)
        return {
            "loss": self.loss_metrics.result(),
            "word_loss": self.word_loss_metric.result(),
            "deep_loss": self.deep_loss_metric.result(),
            "word_acc": self.word_acc_metric.result()
        }

    def test_step(self, data):
        # unpack the data, input has two pair of audio and y has one word reference
        x, word_reference = data
        # compute the loss for each pair
        pair_data = self.compute_pair(x, training=False)
        avg_word_loss, deep_loss = self.compute_loss_pair(pair_data, word_reference)
        # compute the total loss
        total_loss = avg_word_loss + deep_loss
        # update the metrics
        (student_output, _), (reference_output, _) = pair_data
        self.loss_metrics.update_state(total_loss)
        self.word_loss_metric.update_state(avg_word_loss)
        self.deep_loss_metric.update_state(deep_loss)
        self.word_acc_metric.update_state(word_reference.flat_values, student_output.flat_values)
        self.word_acc_metric.update_state(word_reference.flat_values, reference_output.flat_values)
        return {
            "loss": self.loss_metrics.result(),
            "word_loss": self.word_loss_metric.result(),
            "deep_loss": self.deep_loss_metric.result(),
            "word_acc": self.word_acc_metric.result()
        }

    # define metrics
    @property
    def metrics(self):
        return [self.loss_metrics, self.word_loss_metric, self.deep_loss_metric, self.word_acc_metric]
