from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
import argparse
import sys
import time
from util_function import init_tensorboard, path_resolve, EmergencyExit, EmergencyExitCallback
from DeepFeature_DataSet.Test_DS_Factory_Siri import Prosody_Data_Generation, Test_DS_Factory_Siri
from pathlib import Path
from shutil import rmtree
import numpy as np
import scipy


class FrameFeatureNetwork(tf.keras.Model):
    def __init__(self, model_config, restore_path):
        """
        :param model_config: model config
        :param restore_path: a path has the checkpoint /prefix/checkpoint/{epoch:06d}_{val_loss:.2f}.ckpt, it's a string
        """
        super(FrameFeatureNetwork, self).__init__()
        self.network = ASR_Network(**model_config)
        # cleanup the restore path
        restore_path = Path(restore_path).parent
        # Get latest checkpoint from the restore path
        ckpt_path = tf.train.latest_checkpoint(restore_path)
        # Restore the latest checkpoint
        self.network.load_weights(ckpt_path)

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None, mask=None):
        audio_frame = inputs
        base_output, maps = self.network.base_network(audio_frame, training=training)
        return base_output


if __name__ == "__main__":
    """
    This script is used to generate the frame feature for Siri and student voice
    it contains two parts:
    1. generate the frame feature for Siri voice
    2. generate the frame feature for student prosody data
    
    the configure file is config.yaml
    it contains the following information:
    1. model_setting: the model setting for the network
    2. model_storage: the model storage for the network
    3. siri_data_setting: the data setting for the Siri voice
        - frame_feature_path: the path which is dir for the frame feature
        - segment_feature_path: the path which is dir for the segment feature
        - output_path: the path which is dir for the output feature
    4. prosody_data_location: the data setting for the student voice
        - input_path: dir to the wav audio file
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
    network = FrameFeatureNetwork(config['model_setting'], config['model_storage']['model_ckpt'])
    # 1. generate the frame feature for Siri voice
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

    # 2. generate the frame feature for student voice
    # create the directory for the student output file if the directory is existed, prune it
    output_path = Path(config['prosody_data_location']['output_path'])
    if output_path.exists():
        rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Start generating the mfcc feature for student voice on prosody data")
    # read all wave and save it into a serialized tensor on the disk using python runtime
    path_wav = Path(config['prosody_data_location']['input_path'])
    path_tfs = Path(config['prosody_data_location']['tfs_path'])
    if path_tfs.exists():
        rmtree(path_tfs)
    path_tfs.mkdir(parents=True, exist_ok=True)
    # handle the exception related to file cannot read or wav is not valid print file that somehow failed ignore it
    # the channel is always 1, save it the shape need to be (None) instead of (None,1)
    for wav_file in path_wav.glob('*.wav'):
        try:
            audio = tf.io.read_file(str(wav_file))
            audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
            # resample the audio to 16k using scipy resample poly
            audio_np = audio[:, 0].numpy()
            audio_np = scipy.signal.resample_poly(audio_np, 1, 3)
            audio = tf.convert_to_tensor(audio_np, dtype=tf.float32)
            mfcc = Prosody_Data_Generation.get_mfcc(audio, is_float=True)
            d_serialized = tf.io.serialize_tensor(mfcc)
            tf.io.write_file(str(path_tfs / f'{wav_file.stem}.tfs'),
                             d_serialized)
            print(f"Finish processing file {wav_file}")
        except Exception as e:
            print(f"Error reading file {wav_file}, error: {e}")
            continue

    # create the dataset for the Prosody voice
    student_ds_factory = Prosody_Data_Generation(audios_wav_path=config['prosody_data_location']['tfs_path'])
    student_ds = student_ds_factory.get_final_ds(batch_size=4)

    # iterate though siri dataset and save the frame feature using name
    print("Start generating the frame feature for siri voice")
    for (audio, (start, duration)), passage_id, record_index in siri_ds:
        frame_feature = network(audio)
        # save the frame feature as a tensor to disk
        d_serialized = tf.io.serialize_tensor(frame_feature)
        # save to disk using passage_id.tfs, passage_id is a int64
        tf.io.write_file(str(Path(config['siri_data_setting']['output_path']) / f'{passage_id.numpy()}.tfs'),
                         d_serialized)
    print("Siri frame feature is generated")

    print("Start generating the frame feature for student voice prosody data")
    # iterate though student dataset and save the frame feature using name
    # the batch size is 4, save it using name
    for record in student_ds:
        frame_feature = network(record['mfcc'])
        # save the frame feature as a tensor to disk
        # iterate though the batch and save the frame feature, cut each record using mask
        # if mask is 0 it means the frame is padding, we don't need to save it
        for i in range(frame_feature.shape[0]):
            bool_mask = tf.equal(record['mask'][i],1)
            final_frame_feature = tf.boolean_mask(frame_feature[i], bool_mask)
            if tf.size(final_frame_feature) == 0:
                print(f"Skipping empty feature for sample {record['file_name'][i].numpy().decode('utf-8')}")
                continue
            d_serialized = tf.io.serialize_tensor(final_frame_feature)
            # save to disk using record['file_name'][i].tfs, record['file_name'][i] is tf.string
            tf.io.write_file(str(Path(config['prosody_data_location']['output_path']) / f'{record["file_name"][i].numpy().decode("utf-8")}.tfs'),
                             d_serialized)
