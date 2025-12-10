import sys

sys.path.append('..')
from ASR_Network import ASR_Network
import tensorflow as tf
from pathlib import Path
import argparse
from yaml import safe_load


class FrameGenerateion(tf.keras.Model):
    def __init__(self, model_asr_config, restore_path_asr):
        super(FrameGenerateion, self).__init__()
        model_asr_config['batch_num'] = 1
        self.asr_model = ASR_Network(**model_asr_config)
        restore_path = Path(restore_path_asr)
        ckpt_path = tf.train.latest_checkpoint(restore_path)
        self.asr_model.load_weights(ckpt_path)

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None, mask=None):
        frame, _ = self.asr_model.base_network(inputs, training=training, mask=mask)
        return frame


def map_ds(file_path):
    # Load the tfs file using tf.io.read_file
    tfs = tf.io.read_file(file_path)
    # Decode the tfs file
    mfc = tf.io.parse_tensor(tfs, out_type=tf.float32)
    # mfc frame has shape [4, time, 80]

    # get file name from the file path
    file_name = tf.strings.split(file_path, '/')[-1]
    return mfc, file_name


if __name__ == "__main__":
    # load the config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        main_config = safe_load(f)
    # Load ASR model config file defined in the main config file
    with open(main_config['model_asr_config'], 'r') as f:
        model_asr_config = safe_load(f)

    # Get a Store path for the ASR model
    restore_path_asr = Path(model_asr_config['model_storage']['model_ckpt']).parent
    # Create the network
    network = FrameGenerateion(model_asr_config["model_setting"], restore_path_asr)

    # Generate the DataSet Load using tf.data.Dataset and glob pattern

    ds = tf.data.Dataset.list_files(str(Path(main_config['data']['input_path'])) + '/*.tfs')
    ds = ds.map(map_ds)

    # iterate over the dataset and generate the frame feature and save to disk
    output_path = Path(main_config['data']['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    for mfc, file_name in ds:
        frame = network(mfc)
        file_name = file_name.numpy().decode()
        tf.io.write_file(str(output_path / file_name), tf.io.serialize_tensor(frame))
        print(f"Frame feature for {file_name} generated and saved to {output_path}")

