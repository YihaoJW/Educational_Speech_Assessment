from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
import argparse
from pathlib import Path
import numpy as np


class DeepFeatureNetwork(tf.keras.Model):
    def __init__(self, model_config, restore_path):
        """
        :param model_config: model config
        :param restore_path: a path has the checkpoint /prefix/checkpoint/{epoch:06d}_{val_loss:.2f}.ckpt, it's a string
        """
        super(DeepFeatureNetwork, self).__init__()
        self.network: ASR_Network = ASR_Network(**model_config)
        # cleanup the restore path
        restore_path: Path = Path(restore_path).parent
        # Get latest checkpoint from the restore path
        ckpt_path = tf.train.latest_checkpoint(restore_path)
        # Restore the latest checkpoint
        self.network.load_weights(ckpt_path)

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None, mask=None):
        audio, (start, duration) = inputs
        base_output, maps = self.network.base_network(audio, training=training)
        total_maps = [base_output] + maps
        pooled_maps = self.network.pooling(total_maps, start, duration)
        deep_feature = tf.ragged.map_flat_values(lambda x: self.network.deep_feature(x, training=training, mask=mask),
                                                 pooled_maps)
        return deep_feature


def compute_batch_jacobin(word_prediction_network, experiment_count, mean=0.0, variance=1.0):
    # Calculate standard deviation from variance
    stddev = tf.math.sqrt(variance)

    # Generate a batch of X with the given mean and stddev
    X = tf.random.normal([experiment_count, 128], mean=mean, stddev=stddev)

    # Create an empty TensorArray to hold the results
    results = tf.TensorArray(dtype=tf.float32, size=experiment_count)

    for i in range(experiment_count):
        # Indexing individual x using i and adding new axis to make its shape [1, 128]
        x = X[i][tf.newaxis, :]

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = word_prediction_network(x)

        # Compute the Jacobin and write it to the TensorArray
        results = results.write(i, tf.squeeze(tape.jacobian(y, x)))

    # Stack the TensorArray results
    stacked_results = results.stack()

    # Average over the first axis
    averaged_results = tf.reduce_mean(stacked_results, axis=0)

    return averaged_results


if __name__ == "__main__":
    """
    This script generate the embedding of words using gradient of the decoder network
    the configure file is config.yaml
    it contains the following information:
    1. model_setting: the model setting for the network
    2. model_storage: the model storage for the network
    
    the configure file is config.yaml
    it contains the following information:
    1. model_setting: the model setting for the network
    2. model_storage: the model storage for the network
    3. siri_data_setting: the data setting for the Siri voice
        - frame_feature_path: the path which is dir for the frame feature
        - segment_feature_path: the path which is dir for the segment feature
        - output_path: the path which is dir for the output feature
    4. student_data_setting: the data setting for the student voice
        - frame_feature_path: the path which is a tensorflow dataset record for the frame feature
        - segment_feature_path: the path which is a dir for the segment feature
        - output_path: the path which is a dir for the output feature
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--experiment_count', type=int, default=512)
    parser.add_argument('--output', type=str, default='embedding.npy')
    args = parser.parse_args()
    # load the config
    with open(args.config, 'r') as f:
        config = safe_load(f)
    # bath num is for loss calculation, it will not use in the prediction, set it to 1 as placeholder
    config['model_setting']['batch_num'] = 1
    # load the network
    network = DeepFeatureNetwork(config['model_setting'], config['model_storage']['model_ckpt'])
    decoder = network.network.word_prediction
    # generate the embedding
    embedding = compute_batch_jacobin(decoder, args.experiment_count)

    # save the embedding
    # get prefix for saving the embedding
    prefix = Path(config['model_storage']['model_restore']).parent.parent
    # save use TSV format prefix/ and args.output
    np.savetxt(prefix / args.output, embedding, delimiter='\t')
