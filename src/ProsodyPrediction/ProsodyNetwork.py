import sys

sys.path.append('..')
from AttentionModule import SelfAttention, SwiGLU, RotaryEmbeddingMask, CrossAttention
import tensorflow as tf


def residual_block(x, channels, filter_size, dropout_rate=-1.0, attention_heads=2):
    x_input = x
    x = tf.keras.layers.LayerNormalization()(x)
    x = SwiGLU()(x)
    x = tf.keras.layers.SeparableConv1D(channels, filter_size, padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = SwiGLU()(x)
    x = tf.keras.layers.SeparableConv1D(channels, filter_size, padding='same')(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return tf.keras.layers.Add()([x_input, x])


def residual_block_stack(x, channels, filter_size, stack_size, dropout_rate=-1.0, attention_heads=2):
    x = tf.keras.layers.Conv1D(channels, filter_size, padding='same')(x)
    for i in range(stack_size):
        x = residual_block(x, channels, filter_size, dropout_rate, attention_heads)
    return x


def build_down_sample(x, output_shape, channels_list, filter_size, stack_size, dropout_rate=-1.0, attention_heads=2):
    # Build the encoder
    for i in range(len(channels_list)):
        x = residual_block_stack(x, channels_list[i], filter_size, stack_size, dropout_rate, attention_heads)
        if i < len(channels_list) - 1:
            x = tf.keras.layers.SeparableConv1D(channels_list[i], 2, strides=2)(x)
    x = tf.keras.layers.Conv1D(output_shape, 1, padding='same')(x)
    return x


def create_base_network(input_shape,
                        feature_depth,
                        channels_list,
                        filter_size,
                        stack_size,
                        dropout_rate=-1.0,
                        attention_heads=2):
    x = tf.keras.Input(input_shape)
    y = tf.keras.layers.Masking(mask_value=-1.0)(x)
    y = tf.keras.layers.Dense(channels_list[0])(y)
    y = RotaryEmbeddingMask()(y)
    y = build_down_sample(y, feature_depth, channels_list, filter_size, stack_size,
                          dropout_rate=dropout_rate,
                          attention_heads=attention_heads)
    # Generate mask for the output
    return tf.keras.Model(x, y)


def residual_self_attention_block(x, channels, filter_size, dropout_rate=-1.0, attention_heads=2):
    x = SelfAttention(num_heads=attention_heads, key_dim=channels,
                      dropout=dropout_rate if dropout_rate > 0 else 0.0)(x)
    x_resi = x
    x = tf.keras.layers.Dense(channels * 2)(x)
    x = SwiGLU()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([x_resi, x])
    return tf.keras.layers.LayerNormalization()(x)


def residual_self_attention_block_stack(x, channels, filter_size, stack_size, dropout_rate=-1.0, attention_heads=2):
    if x.shape[-1] != channels:
        x = tf.keras.layers.Dense(channels)(x)
    for i in range(stack_size):
        x = residual_self_attention_block(x, channels, filter_size, dropout_rate, attention_heads)
    return x


# Create Self-Attention Network, Post-Processing is not included it will include before base network
def create_self_attention_network(input_shape,
                                  channels_list,
                                  filter_size,
                                  stack_size,
                                  dropout_rate=-1.0,
                                  attention_heads=2):
    x = tf.keras.Input(input_shape)
    y = x
    for channels in channels_list:
        y = residual_self_attention_block_stack(y, channels, filter_size, stack_size,
                                                dropout_rate=dropout_rate,
                                                attention_heads=attention_heads)
    return tf.keras.Model(x, y)


# Generate a Cross Attention block, it need two input x and y, x is the query, y is the key and value
def residual_cross_attention_block(x, y, channels, filter_size, dropout_rate=-1.0, attention_heads=2):
    x = CrossAttention(num_heads=attention_heads, key_dim=channels,
                       dropout=dropout_rate if dropout_rate > 0 else 0.0)([x, y])
    x_resi = x
    x = tf.keras.layers.Dense(channels * 2)(x)
    x = SwiGLU()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([x_resi, x])
    return tf.keras.layers.LayerNormalization()(x)


def residual_cross_attention_block_stack(x, y, channels, filter_size, stack_size, dropout_rate=-1.0, attention_heads=2):
    # if x has the same feature as channels, we can skip the first layer (skip Linear Adjustment Layer)
    if x.shape[-1] != channels:
        x = tf.keras.layers.Dense(channels)(x)
    if y.shape[-1] != channels:
        y = tf.keras.layers.Dense(y)
    for i in range(stack_size):
        x = residual_self_attention_block(x, channels, filter_size, dropout_rate, attention_heads)
        x = residual_cross_attention_block(x, y, channels, filter_size, dropout_rate, attention_heads)
    return x


# Create Cross-Attention Network, Post-Processing is not included it will include before base network
def create_cross_attention_network(input_shape,
                                   channels_list,
                                   filter_size,
                                   stack_size,
                                   dropout_rate=-1.0,
                                   attention_heads=2):
    x = tf.keras.Input(input_shape)
    cross_info = tf.keras.Input(input_shape)
    y = x
    for channels in channels_list:
        y = residual_cross_attention_block_stack(y, cross_info, channels, filter_size, stack_size,
                                                 dropout_rate=dropout_rate,
                                                 attention_heads=attention_heads)
    return tf.keras.Model([x, cross_info], y)


class ProsodyNetwork(tf.keras.Model):
    def __init__(self, base_feature, attention_feature, attention_feature_output, output_sz, **kwargs):
        # base_feature need contain key drop_out_rate, attention_heads
        assert 'dropout_rate' in base_feature, "dropout_rate is not in the base_feature"
        assert 'attention_heads' in base_feature, "attention_heads is not in the base_feature"
        super().__init__(**kwargs)
        self.down_sample_network = create_base_network(**base_feature)
        self.positional_embedding = RotaryEmbeddingMask()
        self.self_attention = create_self_attention_network(**attention_feature)
        self.cross_attention = create_cross_attention_network(**attention_feature)
        self.output_self_attention = create_self_attention_network(**attention_feature_output)
        # final_output global average across the time axis, and reduce to 1
        self.final_output = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(output_sz)
        ])
        self.reduce_factor = 2 ** (len(base_feature['channels_list']) - 1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        student_voice, siri_voices = inputs
        masks = self.calculate_down_sample_mask(inputs)
        student_voice_mask, siri_voices_mask = masks
        # student_voicem and siri_voice need the input for the base network siri voices are a list
        student_voice_frame_feature = self.down_sample_network(student_voice, training=training, mask=mask)
        siri_voices_frame_feature = [self.down_sample_network(x, training=training, mask=mask) for x in siri_voices]
        # Apply the positional embedding again
        student_voice_frame_feature = self.positional_embedding(student_voice_frame_feature,
                                                                training=training,
                                                                mask=student_voice_mask)

        siri_voices_frame_feature = [self.positional_embedding(x, training=training, mask=m) for x, m in
                                     zip(siri_voices_frame_feature, siri_voices_mask)]
        # self-attention apply to and siri voice
        siri_voices_atten = [self.self_attention(x, training=training) for x in siri_voices_frame_feature]
        # cross-attention apply between student voice and siri voices
        student_voice_cross_atten = [self.cross_attention([student_voice_frame_feature, x],
                                                          training=training) for x in siri_voices_atten]
        # concat the cross-attention output
        student_cross_concat = tf.concat(student_voice_cross_atten, axis=-1)
        # output self-attention
        student_voice_output = self.output_self_attention(student_cross_concat,
                                                          training=training, mask=student_voice_mask)
        # final output
        return self.final_output(student_voice_output, training=training)

    def calculate_mask_single(self, input_tensor):
        """
        Calculate and downsample the mask for a single input tensor.
        """
        # Calculate the mask for the input tensor
        mask = tf.reduce_any(input_tensor != -1.0, axis=-1, keepdims=True)
        # Downsample the mask
        mask_downsampled = tf.nn.max_pool1d(tf.cast(mask, tf.float32), ksize=self.reduce_factor,
                                            strides=self.reduce_factor, padding='VALID')
        mask_downsampled = tf.reduce_any(mask_downsampled > 0.0, axis=-1)
        return mask_downsampled

    def calculate_down_sample_mask(self, inputs):
        student_voice, siri_voices = inputs

        # Use calculate_mask_single for the student voice
        student_mask_downsampled = self.calculate_mask_single(student_voice)

        # Calculate and downsample the mask for each siri voice using calculate_mask_single
        siri_masks_downsampled = [self.calculate_mask_single(x) for x in siri_voices]

        return student_mask_downsampled, siri_masks_downsampled

    def get_config(self):
        config = super().get_config()
        return config

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # if y_pred is nan we need to skip the update
            y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # if loss is nan we need to skip the update
            loss = tf.where(tf.math.is_finite(loss), loss, 0.0)
        trainable_vars = self.trainable_variables
        self._validate_target_and_loss(y, loss)
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
