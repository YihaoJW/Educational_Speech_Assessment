import time
from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras import layers, activations
from shutil import rmtree
import sys, os, subprocess, yaml
from datetime import datetime, timedelta


def generate_sprite_image(image_list, image_size):
    # Create a TensorFlow dataset from the image list
    dataset = tf.data.Dataset.from_tensor_slices(image_list)
    # Load, decode, and resize the images
    images = dataset.map(lambda path: tf.image.resize(tf.image.decode_image(tf.io.read_file(path)), image_size))
    # Stack all images into a single tensor
    stacked_images = tf.stack(list(images), axis=0)
    # Compute the sprite size
    sprite_size = int(tf.math.ceil(tf.math.sqrt(len(image_list))))
    # Compute the sprite shape
    sprite_shape = (sprite_size * image_size[0], sprite_size * image_size[1], 3)
    # Create an empty sprite image
    sprite_image = tf.zeros(sprite_shape, dtype=tf.uint8)
    # Calculate the indices for all images without using loops
    rows, cols = tf.meshgrid(tf.range(0, sprite_size * image_size[0], image_size[0]),
                             tf.range(0, sprite_size * image_size[1], image_size[1]),
                             indexing='ij')
    flat_rows = tf.reshape(rows, [-1])[:len(image_list)]
    flat_cols = tf.reshape(cols, [-1])[:len(image_list)]
    indices = tf.stack([flat_rows, flat_cols], axis=-1)
    # Update the sprite image with all images at the same time
    sprite_image = tf.tensor_scatter_nd_update(sprite_image, indices, stacked_images)
    return sprite_image


def init_tensorboard(log_dir, name=None):
    """
    initialize tensorboard session
    :param name: Name for Tensorboard name
    :param log_dir:
    :return:
    """
    # check the log_dir not exist init a new one
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    # if name is not given use 'ASR_Voice_' + str(log_dir.parent.parent.name) else use ASR_Voice_ + name
    if name is None:
        name = 'ASR_Voice_' + str(log_dir.parent.parent.name)
    else:
        name = 'ASR_Voice_' + str(name)

    # add time in to name
    describe = str(name) + '_' + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    command = f"tensorboard dev upload --logdir {str(log_dir)} --name {str(name)} --description {str(describe)} --verbose 0"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # wait for the subprocess to start
    return process


# helper function resolve path related issue it receives a dict return void
def path_resolve(config_dict, args):
    """
    Sample of configuration file
    config = {'model_setting': {'base_feature': dict(zip(base_feature_name, base_feature)),
                            'dense_feature': dict(zip(dense_feature_name, dense_feature)),
                            'word_prediction': dict(zip(word_prediction_name, word_prediction)),
                            'base_ratio': base_ratio
                            'margin': 0.4},
          'model_storage': {'model_ckpt': 'checkpoint/{epoch:06d}_{val_loss:.2f}.ckpt',
                            'model_restore': 'backup/model.ckpt',
                            'tensorboard_path': 'tensorboard/'},
          'training_setting': {'batch_size': 32,
                               'epoch': 1000,
                               'learning_rate': {'initial': 0.001,
                                                 'decay': 0.1,
                                                 'decay_step': 100},
                               },
          'data_location': {'data_record': 'Tensorflow_DataRecord/Student_Answer_Record.tfrecord',
                            'siri_voice': 'Siri_Related/Siri_Reference_Sample',
                            'siri_meta': 'Siri_Related/Siri_Dense_Index'},
          'cache_location': {'cache': 'cache/'}
          }
    This function will generate all folders if they don't exist. If retrain is True, it will delete the old model.
    After this function is called, the dir is guaranteed to exist, and the folder is ready for resume or retrain.
    """
    for key in config_dict['data_location']:
        config_dict['data_location'][key] = Path(config_dict['data_location'][key])
        if not config_dict['data_location'][key].exists():
            raise FileNotFoundError(f'{key} not exist')
    # for cache if not exist create the parent folder
    for key in config_dict['cache_location']:
        config_dict['cache_location'][key] = Path(config_dict['cache_location'][key])
        if not config_dict['cache_location'][key].parent.is_dir():
            config_dict['cache_location'][key].parent.mkdir(parents=True, exist_ok=True)
    # for model storage if not exist create it
    for key in config_dict['model_storage']:
        config_dict['model_storage'][key] = Path(config_dict['model_storage'][key])
        if not config_dict['model_storage'][key].parent.is_dir():
            config_dict['model_storage'][key].parent.mkdir(parents=True, exist_ok=True)
        # if retrain is true delete the old model and create the folder
        if args.retrain:
            if config_dict['model_storage'][key].is_dir():
                if config_dict['model_storage'][key].is_file():
                    config_dict['model_storage'][key].unlink()
                else:
                    rmtree(config_dict['model_storage'][key])
            else:
                # if the path is not a dir
                if config_dict['model_storage'][key].is_file():
                    config_dict['model_storage'][key].unlink()
            # make dir and it's parent if exist do nothing
            config_dict['model_storage'][key].mkdir(parents=True, exist_ok=True)


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

    def call(self, x, training=False, mask=None):
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


class EmergencyExit(Exception):
    """Exception raised for emergency exit in the program."""

    def __init__(self, message="Emergency exit triggered."):
        self.message = message
        super().__init__(self.message)


# Define a callback that check if EmergencyExit is need to be triggered
# the init method receive a remain time in minutes, and variable name for the time
# End time can be read using environment variable default from SLURM, it only checks on training batch start
class EmergencyExitCallback(tf.keras.callbacks.Callback):
    def __init__(self, remain_time=45, time_name="JOB_END_TIME", *args, **kwargs):
        super().__init__()
        self.remain_time = remain_time
        self.time_name = time_name
        self.time_limit_str = os.environ.get(self.time_name, None)
        self.log_time_start = datetime.now()
        self.epoch_time = self.remain_time
        if self.time_limit_str is not None:
            self.time_limit = datetime.strptime(self.time_limit_str, "%Y-%m-%dT%H:%M:%S")
            print(f" Default Time limit is set to {self.time_limit}")
        else:
            self.time_limit = None

    def on_epoch_begin(self, epoch, logs=None):
        # log the time
        if self.time_limit is not None:
            self.log_time_start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Estimate the epoch time
        epoch_wall_time = datetime.now() - self.log_time_start
        # Update epoch time using Exponential Moving Average with factor 0.1, epoch time is in minutes
        self.epoch_time = int(self.remain_time * 0.9 + (epoch_wall_time.total_seconds() / 60) * 0.1)
        # add a margin of 10 minutes and set it as the remaining time
        self.remain_time = self.epoch_time + 10
        if self.time_limit is not None:
            current_time = datetime.now()
            remaining_time = self.time_limit - current_time
            if remaining_time < timedelta(minutes=self.remain_time):
                raise EmergencyExit(f"Exiting because there lack of time to finish the epoch: {self.remain_time}.")
            else:
                print(f" Remaining time: {remaining_time}")


def load_config(file_path):
    # Read the YAML file into a string
    with open(file_path, 'r') as f:
        config_str = f.read()

    # Replace environment variables using echo in a shell
    config_str = subprocess.check_output(['bash', '-c', f"echo '{config_str}'"]).decode()

    # Parse the resulting string back into a Python dict
    config = yaml.safe_load(config_str)

    return config
