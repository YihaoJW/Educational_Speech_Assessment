import tensorflow as tf
from tensorflow.python.keras import layers, activations


@tf.function
def inform_pooling(value, start, duration, ratio):
    batch = tf.shape(value)[0]
    end = start + duration
    start = tf.math.floor(start * ratio)
    end = tf.math.ceil((end + 0.001) * ratio)

    period = tf.cast(tf.stack([start, end], axis=-1), tf.int32)
    tf.debugging.assert_less(period[..., 0], period[..., 1])
    ret_b = tf.TensorArray(tf.float32, batch, infer_shape=False)
    ret_count = tf.TensorArray(tf.int32, batch)
    for batch_index in tf.range(batch):
        value_l = value[batch_index]
        val_ind_max = tf.shape(value_l)[0]
        period_l = period[batch_index]
        period_l_p = tf.math.minimum(period_l, val_ind_max - 1)
        ret_count = ret_count.write(batch_index, tf.shape(period_l)[0])
        indexes = tf.ragged.range(period_l_p[..., 0], period_l_p[..., 1])
        value_indices = tf.gather(value_l, indexes)
        pooled = tf.reduce_mean(value_indices, axis=1)
        ret_b = ret_b.write(batch_index, pooled)
    row_length = ret_count.stack()
    ret = ret_b.concat()
    return tf.RaggedTensor.from_row_lengths(ret, row_length)


@tf.function
def Get_Gradient(value, start, duration, ratio, A):
    with tf.GradientTape() as t:
        v2 = value * A
        Final_Tensor = inform_pooling(v2, start, duration, ratio)
        J = tf.reduce_mean(Final_Tensor)
        # tf.print(f'Losses: {J}, Output Shape: {tf.shape(Final_Tensor)}, Input Shape{tf.shape(v2)}')
        G = t.gradient(J, [A])
    return G


class GatedXVector(tf.keras.layers.Layer):
    @tf.function()
    def call(self, input):
        x = input[0]
        x_at = input[1]
        x = tf.cast(x, self.dtype_policy.variable_dtype)
        x_at = tf.cast(x_at, self.dtype_policy.variable_dtype)
        gated = x * x_at
        return tf.cast(tf.reduce_sum(gated, axis=[-2]), self.dtype_policy.compute_dtype)


class PositionEncoding1D(layers.Layer):
    def __init__(self, default_position=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.built = None
        self.pos_encoding = None
        self.d_model = None
        self.def_position = default_position
        with tf.init_scope():
            self.position = self.def_position

    def get_config(self):
        config = super().get_config()
        config.update({"def_position": self.def_position})
        return config

    def build(self, input_shape):
        assert input_shape[-1] is not None, "Unit cannot undefined"
        self.d_model = input_shape[-1]
        if input_shape[-2] is None:
            self.position = self.def_position
        else:
            self.position = input_shape[-2]
        self.pos_encoding = self.positional_encoding()
        self.built = True

    def positional_encoding(self):
        angle_rates = 1 / tf.pow(10000.,
                                 tf.cast(tf.range(0, self.d_model, 2), self.dtype_policy.variable_dtype) / tf.cast(
                                     self.d_model, self.dtype_policy.variable_dtype))
        angle_rads = tf.einsum('i,j->ij', tf.cast(tf.range(self.position), self.dtype_policy.variable_dtype),
                               angle_rates)
        sin_cos = tf.math.sin(angle_rads)[..., tf.newaxis], tf.math.cos(angle_rads)[..., tf.newaxis]
        pos_encoding = tf.reshape(tf.concat(sin_cos, axis=-1), [angle_rads.shape[0], -1])[:, :self.d_model]
        return pos_encoding

    def get_encode(self, length, x, tig=3):
        encoder_msg = tf.cast(self.pos_encoding[0: length][tf.newaxis, ...], self.dtype_policy.compute_dtype)
        try:
            encode = encoder_msg + x
        except tf.errors.InvalidArgumentError:
            self.position = length if length > self.position * 2 else self.position * 2
            self.pos_encoding = self.positional_encoding()
            assert tig > 0, "Too much iteration, might caused by feature map size change"
            encode = self.get_encode(length, x, tig=tig - 1)
        return encode

    def call(self, x, training=False):
        x_length = tf.shape(x)[-2]
        return self.get_encode(x_length, x)


class SelfAttention1DNorm(layers.Layer):
    def __init__(self, QK_size, Value_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.QK_size = QK_size
        self.Value_size = Value_size
        self.Q_Layer = layers.SeparableConv1D(QK_size, (3), padding='same', depth_multiplier=2)
        self.K_Layer = layers.SeparableConv1D(QK_size, (3), padding='same', depth_multiplier=2)
        self.V_Layer = layers.SeparableConv1D(Value_size, (3), padding='same', depth_multiplier=2)
        self.Out_Norm = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({"QK_size": self.QK_size, "Value_size": self.Value_size})
        return config

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def Attention_Merge(self, Q, K, V, training=False):
        Innter_Product = tf.einsum('abd,acd->abc', Q, K) / tf.sqrt(tf.cast(self.QK_size, dtype=Q.dtype))
        C_Innter_Product = tf.cast(Innter_Product, self.dtype_policy.variable_dtype)
        AttentionMap = tf.keras.activations.softmax(C_Innter_Product, [-1])
        Final = tf.einsum('abc, acg->abg', AttentionMap, tf.cast(V, self.dtype_policy.variable_dtype))
        return tf.cast(Final, self.dtype_policy.compute_dtype)

    def call(self, x, training=False):
        Q = self.Q_Layer(x)
        K = self.K_Layer(x)
        V = self.V_Layer(x)

        Out = self.Attention_Merge(Q, K, V, training=training)
        return self.Out_Norm(Out, training=training)


class SelfAttention1DNormReduce(layers.Layer):
    def __init__(self, QK_size, Value_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.QK_size = QK_size
        self.Value_size = Value_size
        self.Q_Layer = layers.SeparableConv1D(QK_size, (3), padding='same', depth_multiplier=2)
        self.K_Layer = layers.SeparableConv1D(QK_size, (3), padding='same', depth_multiplier=2)
        self.V_Layer = layers.SeparableConv1D(Value_size, (3), padding='same', depth_multiplier=2)
        self.Out_Norm = layers.BatchNormalization()

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({"QK_size": self.QK_size, "Value_size": self.Value_size})
        return config

    def attention_merge(self, Q, K, V, training=False):
        Innter_Product = tf.einsum('abd,acd->abc', Q, K) / tf.sqrt(tf.cast(self.QK_size, dtype=Q.dtype))
        C_Innter_Product = tf.cast(Innter_Product, self.dtype_policy.variable_dtype)
        AttentionMap = tf.keras.activations.softmax(C_Innter_Product, [-1, -2])
        Final = tf.einsum('abc, acg->ag', AttentionMap, tf.cast(V, self.dtype_policy.variable_dtype))
        return tf.cast(Final, self.dtype_policy.compute_dtype)

    def call(self, x, training=False):
        Q = self.Q_Layer(x)
        K = self.K_Layer(x)
        V = self.V_Layer(x)

        #     Qn = self.Q_Norm(Q, training = training)
        #     Kn = self.K_Norm(K, training = training)
        #     Vn = self.V_Norm(V, training = training)

        Out = self.attention_merge(Q, K, V, training=training)
        return self.Out_Norm(Out, training=training)
