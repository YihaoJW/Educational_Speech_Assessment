import tensorflow as tf
from AttentionModule import CutConcatenate, CrossAttention, SelfAttention, RotaryEmbeddingMask


# A function that generates a residual Block using separable convolution
def residual_block(x, channels, filter_size, dropout_rate=-1.0, attention_heads=2):
    x_input = x
    # Self-attention
    if channels > 192:
        x = SelfAttention(num_heads=attention_heads, key_dim=channels,
                          dropout=dropout_rate if dropout_rate > 0 else 0.0)(x)
    else:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv1D(channels, filter_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv1D(channels, filter_size, padding='same')(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    return tf.keras.layers.Add()([x_input, x])


# Build a Convolutional with to adjust the input to the residual block and apply several residual block
def residual_block_stack(x, channels, filter_size, stack_size, dropout_rate=-1.0, attention_heads=2):
    x = tf.keras.layers.Conv1D(channels, filter_size, padding='same')(x)
    for i in range(stack_size):
        x = residual_block(x, channels, filter_size, dropout_rate, attention_heads)
    return x


# Function builds a U-Net
def build_unet(x, output_shape, channels_list, filter_size, stack_size, dropout_rate=-1.0, attention_heads=2):
    # Build the encoder
    encoder = []
    decoder = []
    for i in range(len(channels_list)):
        x = residual_block_stack(x, channels_list[i], filter_size, stack_size, dropout_rate, attention_heads)
        encoder.append(x)
        if i < len(channels_list) - 1:
            x = tf.keras.layers.AvgPool1D(2)(x)
    # Build the decoder
    for i in range(len(channels_list) - 2, -1, -1):
        # Stride 2 convolution to upsample
        x = tf.keras.layers.Conv1DTranspose(channels_list[i], 4, strides=2, padding='valid')(x)
        # x = CutConcatenate(axis=-1)([encoder[i], x])
        if channels_list[i] > 192:
            x_c = CrossAttention(num_heads=attention_heads, key_dim=channels_list[i], dropout=dropout_rate)([x, encoder[i]])
            x = tf.keras.layers.Concatenate(axis=-1)([x, x_c])
        else:
            x = CutConcatenate(axis=-1)([encoder[i], x])
        x = residual_block_stack(x, channels_list[i], filter_size, stack_size, dropout_rate)
        decoder.append(x)
    # Build the output
    x = tf.keras.layers.Conv1D(output_shape, 1, padding='same')(x)
    return x, decoder


# Build a Network
def build_network(input_shape, output_shape, channels_list, filter_size, stack_size, dropout_rate=-1.0):
    x = tf.keras.Input(input_shape)
    y, maps = build_unet(x, output_shape, channels_list, filter_size, stack_size, dropout_rate)
    return tf.keras.Model(x, y)


# Define a residual block that uses a fully connected layer
def residual_block_fc(x, channels, dropout_rate=-1.):
    x_input = x
    # Residual block start Normalizes, Activates, and Convolution
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(channels)(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(channels)(x)

    return tf.keras.layers.Add()([x_input, x])


# %%
# Build a stack for fully connected layer
def residual_block_stack_fc(x, channels, stack_size, dropout_rate=-1.):
    x = tf.keras.layers.Dense(channels)(x)
    for i in range(stack_size):
        x = residual_block_fc(x, channels, dropout_rate)
    return x


# Define a min max clip constraint
class MinMaxClip(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


# Define a Keras layers that perform information pooling
class InformPooling(tf.keras.layers.Layer):
    def __init__(self, num_maps, ratios_list, **kwargs):
        super().__init__(**kwargs)
        self.num_maps_shape = None
        self.num_maps = num_maps
        self.ratios_list = ratios_list

    def build(self, input_shape):
        # Input is a bunch of tensor, calculate the total number of feature maps
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


@tf.function
def transpose_last_two_axes(tensor):
    # Get the rank of the tensor
    rank = len(tensor.shape)

    # Create a permutation list to transpose the last two axes
    perm = list(range(rank - 2)) + [rank - 1, rank - 2]

    # Transpose the last two axes
    transposed_tensor = tf.transpose(tensor, perm)

    return transposed_tensor


class AutoLossBalancing(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auto_balancing = self.add_weight('auto_balancing', shape=(2,), dtype=tf.float32, trainable=True,
                                              initializer='zeros', constraint=MinMaxClip(-2, 2))

    def call(self, inputs, training=None, mask=None):
        word_loss, deep_loss = inputs[0], inputs[1]
        weighted_loss = tf.reduce_sum(
            (1 / (tf.exp(self.auto_balancing)) ** 2) * tf.stack([word_loss, deep_loss / 2], axis=0))
        total_loss = weighted_loss + tf.reduce_sum(self.auto_balancing)
        return total_loss


class ASR_Network(tf.keras.Model):
    def __init__(self,
                 base_feature,
                 dense_feature,
                 word_prediction,
                 base_ratio,
                 batch_num,
                 margin,
                 k_top=5,
                 dropout_rate=0.2,
                 attention_heads=2,
                 **kwargs):

        super().__init__(**kwargs)
        self.base_network = self.create_base_network(attention_heads=attention_heads,
                                                     dropout_rate=dropout_rate,
                                                     **base_feature)
        self.deep_feature = self.build_dense_network(dropout_rate=dropout_rate, **dense_feature)
        self.word_prediction = self.build_dense_network(dropout_rate=dropout_rate, **word_prediction)
        pooling_ratios = [base_ratio / 2 ** i for i in range(len(base_feature['channels_list']))]
        self.pooling = InformPooling(len(pooling_ratios), pooling_ratios)
        # define metrics
        self.loss_metrics = tf.keras.metrics.Mean(name='train_loss')
        self.word_loss_metric = tf.keras.metrics.Mean(name='train_word_loss')
        self.word_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_word_acc')
        self.deep_loss_metric = tf.keras.metrics.Mean(name='train_deep_loss')
        self.k = k_top
        self.top_k_acc_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=self.k,
                                                                               name=f'train_top_{self.k}_word_acc')
        # define loss
        self.category_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.batch_counts = tf.Variable(batch_num, dtype=tf.int64, trainable=False)
        self.margin = margin
        self.auto_balancing_layer = AutoLossBalancing(name='auto_loss_balancing')

    @staticmethod
    def create_base_network(input_shape,
                            feature_depth,
                            channels_list,
                            filter_size,
                            stack_size,
                            dropout_rate=-1.0,
                            attention_heads=2):
        x = tf.keras.Input(input_shape)
        x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        x = RotaryEmbeddingMask()(x)
        y, maps = build_unet(x, feature_depth, channels_list, filter_size, stack_size,
                             dropout_rate=dropout_rate,
                             attention_heads=attention_heads)
        return tf.keras.Model(x, [y, maps])

    @staticmethod
    def build_dense_network(input_shape, output_shape, channels_list, stack_size, dropout_rate=-1.0):
        x = tf.keras.Input(input_shape)
        y = x
        for i in range(len(channels_list)):
            y = residual_block_stack_fc(y, channels_list[i], stack_size, dropout_rate=dropout_rate)
        y = tf.keras.layers.Dense(output_shape)(y)
        return tf.keras.Model(x, y)

    @staticmethod
    @tf.function
    def compute_similarity(value_a, value_b, ref_a, ref_b, margin=0.35, eps=0.01):
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
            ra = ref_a[idx][..., tf.newaxis]
            rb = ref_b[idx][..., tf.newaxis]
            similarity_matrix = tf.matmul(va, vb, transpose_b=True)
            # compute the mask for the positive samples
            raw_mask = tf.equal(ra, transpose_last_two_axes(rb))
            mask = tf.cast(raw_mask, tf.float32)
            # compute the mask for the negative samples
            mask_neg = tf.cast(tf.logical_not(raw_mask), tf.float32)
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

            # Set masked values for positive samples to 1
            min_sim_pos_masked = tf.where(raw_mask, similarity_matrix, 1.0)
            # Set masked values for negative samples to -1
            max_sim_neg_masked = tf.where(tf.math.logical_not(raw_mask), similarity_matrix, -1.0)

            # If raw_mask has any True value, compute the min for positive samples, else return 0
            min_sim_pos = tf.where(tf.math.reduce_any(raw_mask),
                                   tf.reduce_min(min_sim_pos_masked),
                                   tf.constant(0.0, dtype=tf.float32))

            # If the logical NOT of raw_mask has any True value, compute the max for negative samples, else return 0
            max_sim_neg = tf.where(tf.math.reduce_any(tf.math.logical_not(raw_mask)),
                                   tf.reduce_max(max_sim_neg_masked),
                                   tf.constant(0.0, dtype=tf.float32))

            # # compute the max similarity for the positive samples
            # min_sim_pos = tf.reduce_min(tf.multiply(similarity_matrix, mask))
            # # compute the min similarity for the negative samples
            # max_sim_neg = tf.reduce_max(tf.multiply(similarity_matrix, mask_neg))
            # compute the average loss with margin
            loss_avg = tf.maximum(0., margin - avg_sim_pos + avg_sim_neg)
            # compute min_max loss with margin
            loss_min_max = tf.maximum(0., margin - min_sim_pos + max_sim_neg)
            # total loss
            loss = loss_avg + loss_min_max
            loss_array = loss_array.write(idx, loss)
        total_loss = tf.reduce_sum(loss_array.stack())
        return total_loss

    @tf.function
    def call(self, inputs, training=False, mask=None):
        audio, (start, duration) = inputs
        # compute the base network
        base_output, maps = self.base_network(audio, training=training)
        # combine base output and maps
        total_maps = [base_output] + maps
        # pooling the total maps
        pooled_maps = self.pooling(total_maps, start, duration)
        # compute the deep feature
        deep_feature = tf.ragged.map_flat_values(lambda x: self.deep_feature(x, training=training, mask=mask),
                                                 pooled_maps)
        # compute the word prediction
        word_prediction = tf.ragged.map_flat_values(lambda x: self.word_prediction(x, training=training, mask=mask),
                                                    deep_feature)
        return word_prediction, deep_feature

    # compute a input pair
    def compute_pair(self, inputs, training=False):
        student, reference = inputs
        # compute both student and reference
        student_output, student_deep_feature = self(student, training=training)
        reference_output, reference_deep_feature = self(reference, training=training)
        return (student_output, student_deep_feature), (reference_output, reference_deep_feature)

    # get loss for an input pair
    def compute_loss_pair(self, inputs, word_reference):
        (student_output, student_deep_feature), (reference_output, reference_deep_feature) = inputs
        # compute the loss for word prediction
        # Cut value to limited range to avoid crazy value of student_output and reference_output
        student_output = tf.clip_by_value(student_output, -15.0, 15.0)
        reference_output = tf.clip_by_value(reference_output, -15.0, 15.0)
        word_loss_student = self.category_loss(word_reference.flat_values, student_output.flat_values)
        word_loss_reference = self.category_loss(word_reference.flat_values, reference_output.flat_values)
        # Cut avg_word_loss to limited range to avoid crazy value
        word_loss_student = tf.clip_by_value(word_loss_student, 0.0, 60.0)
        word_loss_reference = tf.clip_by_value(word_loss_reference, 0.0, 60.0)

        avg_word_loss = tf.reduce_sum((word_loss_student + word_loss_reference) / 2.) / tf.cast(self.batch_counts,
                                                                                                tf.float32)
        # compute the loss for deep feature
        deep_loss = self.compute_similarity(student_deep_feature, reference_deep_feature, word_reference,
                                            word_reference, margin=self.margin) / tf.cast(self.batch_counts, tf.float32)
        return avg_word_loss, deep_loss

    def train_step(self, data):
        # unpack the data, input has two pairs of audios, and y has one word reference
        x, word_reference = data
        # compute the loss for each pair
        with tf.GradientTape() as tape:
            # compute the loss for each pair
            pair_data = self.compute_pair(x, training=True)
            avg_word_loss, deep_loss = self.compute_loss_pair(pair_data, word_reference)
            # compute the total loss
            total_loss = self.auto_balancing_layer([avg_word_loss, deep_loss], training=True)
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
        # self.word_acc_metric.update_state(word_reference.flat_values, reference_output.flat_values)
        self.top_k_acc_metric.update_state(word_reference.flat_values, student_output.flat_values)
        return {
            "loss": self.loss_metrics.result(),
            "word_loss": self.word_loss_metric.result(),
            "deep_loss": self.deep_loss_metric.result(),
            "word_acc": self.word_acc_metric.result(),
            f"top_{self.k}_word_acc": self.top_k_acc_metric.result()
        }

    def test_step(self, data):
        # unpack the data, input has two pair of audio and y has one word reference
        x, word_reference = data
        # compute the loss for each pair
        pair_data = self.compute_pair(x, training=False)
        avg_word_loss, deep_loss = self.compute_loss_pair(pair_data, word_reference)
        # compute the total loss
        total_loss = self.auto_balancing_layer([avg_word_loss, deep_loss], training=False)
        # update the metrics
        (student_output, _), (reference_output, _) = pair_data
        self.loss_metrics.update_state(total_loss)
        self.word_loss_metric.update_state(avg_word_loss)
        self.deep_loss_metric.update_state(deep_loss)
        self.word_acc_metric.update_state(word_reference.flat_values, student_output.flat_values)
        # self.word_acc_metric.update_state(word_reference.flat_values, reference_output.flat_values)
        self.top_k_acc_metric.update_state(word_reference.flat_values, student_output.flat_values)
        return {
            "loss": self.loss_metrics.result(),
            "word_loss": self.word_loss_metric.result(),
            "deep_loss": self.deep_loss_metric.result(),
            "word_acc": self.word_acc_metric.result(),
            f"top_{self.k}_word_acc": self.top_k_acc_metric.result()
        }

    # define metrics
    @property
    def metrics(self):
        return [self.loss_metrics,
                self.word_loss_metric,
                self.deep_loss_metric,
                self.word_acc_metric,
                self.top_k_acc_metric]
