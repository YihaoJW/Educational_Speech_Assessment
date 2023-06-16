from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
import argparse
import sys
import time
from util_function import init_tensorboard, path_resolve, EmergencyExit, EmergencyExitCallback
from DeepFeature_DataSet.Test_DS_Factory_Siri import Test_DS_Factory_Student, Test_DS_Factory_Siri
from pathlib import Path
from shutil import rmtree


class DeepFeatureNetwork(tf.keras.Model):
    def __init__(self, model_config, restore_path):
        """
        :param model_config: model config
        :param restore_path: a path has the checkpoint /prefix/checkpoint/{epoch:06d}_{val_loss:.2f}.ckpt, it's a string
        """
        super(DeepFeatureNetwork, self).__init__()
        self.network = ASR_Network(**model_config)
        # cleanup the restore path
        restore_path = Path(restore_path).parent
        # Get latest checkpoint from the restore path
        ckpt_path = tf.train.latest_checkpoint(restore_path)
        # Restore the latest checkpoint
        self.network.load_weights(ckpt_path)

    def call(self, inputs, training=None, mask=None):
        audio, (start, duration) = inputs
        base_output, maps = self.network.base_network(audio, training=training)
        total_maps = [base_output] + maps
        pooled_maps = self.network.pooling(total_maps, start, duration)
        deep_feature = tf.ragged.map_flat_values(lambda x: self.network.deep_feature(x, training=training, mask=mask),
                                                 pooled_maps)
        return deep_feature


if __name__ == "__main__":
    """
    This script is used to generate the deep feature for Siri and student voice
    it contains two parts:
    1. generate the deep feature for Siri voice
    2. generate the deep feature for student voice
    
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
        - segment_feature_path: the path which is a dir  for the segment feature
        - output_path: the path which is a dir for the output feature
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    # load the config
    with open(args.config, 'r') as f:
        config = safe_load(f)
    # bath num is for loss calculation, it will not use in the prediction, set it to 1 as placeholder
    config['model_setting']['batch_num'] = 1
    # create the network
    network = DeepFeatureNetwork(config['model_setting'], config['model_storage']['model_ckpt'])

    # 1. generate the deep feature for Siri voice
    # create the directory for the Siri output file if the directory is existed, prune it
    output_path = Path(config['siri_data_setting']['output_path'])
    if output_path.exists():
        rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # create the dataset for the Siri voice
    siri_ds_factory = Test_DS_Factory_Siri(**config['siri_data_setting'])
    siri_ds = siri_ds_factory.get_final_ds()
    # check if dataset works by take first batch, raise exception if it's not work, print the exception
    try:
        (audio, (start, duration)), passage_id, record_index = siri_ds.take(1).get_single_element()
    except Exception as e:
        print(e)
        raise Exception("Siri dataset is not work, please check the dataset")

    # 2. generate the deep feature for student voice
    # create the directory for the student output file if the directory is existed, prune it
    output_path = Path(config['student_data_setting']['output_path'])
    if output_path.exists():
        rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # create the dataset for the student voice
    student_ds_factory = Test_DS_Factory_Student(**config['student_data_setting'])
    student_ds = student_ds_factory.get_final_ds()
    # check if dataset works by take first batch, raise exception if it's not work
    try:
        (audio, (start, duration)), passage_id, record_index = student_ds.take(1).get_single_element()
    except Exception as e:
        print(e)
        raise Exception("Student dataset is not work, please check the dataset")

    # iterate though siri dataset and save the deep feature using name
    for (audio, (start, duration)), passage_id, record_index in siri_ds:
        d_feature = network((audio, (start, duration))).to_tensor(default_value=-1000.)
        # save the deep feature as a tensor to disk
        d_serialized = tf.io.serialize_tensor(d_feature)
        # save to disk using passage_id.tfs, passage_id is a int64
        tf.io.write_file(str(Path(config['siri_data_setting']['output_path']) / f'{passage_id.numpy()}.tfs'), d_serialized)

    # iterate though student dataset and save the deep feature using name
    # the batch size is 1, save it using record_index which is an array of tf.string shape (1,)
    for (audio, (start, duration)), passage_id, record_index in student_ds:
        d_feature = network((audio, (start, duration))).to_tensor(default_value=-1000.)
        # save the deep feature as a tensor to disk
        d_serialized = tf.io.serialize_tensor(d_feature)
        # save to disk using record_index.tfs, record_index is a tf.string shape (1,)
        tf.io.write_file(str(Path(config['student_data_setting']['output_path']) / f'{record_index.numpy()[0]}.tfs'), d_serialized)
