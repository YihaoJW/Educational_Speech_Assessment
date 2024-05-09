import sys

sys.path.append('..')
from ASR_Network import ASR_Network
from ProsodyPrediction.ProsodyNetwork import ProsodyNetwork
from yaml import safe_load, dump
import tensorflow as tf
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


class Predict_Prosody_Full(tf.keras.Model):
    def __init__(self, model_asr_config, model_prosody_config, restore_path_asr, restore_path_prosody):
        super(Predict_Prosody_Full, self).__init__()
        model_asr_config['batch_num'] = 1
        self.asr_model = ASR_Network(**model_asr_config)
        restore_path = Path(restore_path_asr)
        ckpt_path = tf.train.latest_checkpoint(restore_path)
        self.asr_model.load_weights(ckpt_path)

        self.prosody_model = ProsodyNetwork(**model_prosody_config)
        restore_path = Path(restore_path_prosody)
        ckpt_path = tf.train.latest_checkpoint(restore_path)
        self.prosody_model.load_weights(ckpt_path)

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=None, mask=None):
        stu_mfcc, siri_frames = inputs
        stu_frame, _ = self.asr_model.base_network(stu_mfcc, training=training, mask=mask)
        new_input = (stu_frame, siri_frames)
        output = self.prosody_model.call(new_input, training=training, mask=mask)
        return output


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='config.yaml')
    args.add_argument('--skip-err', action='store_true')
    args = args.parse_args()
    with open(args.config, 'r') as f:
        main_config = safe_load(f)

    with open(main_config['model_asr_config'], 'r') as f:
        model_asr_config = safe_load(f)
    with open(main_config['model_prosody_config'], 'r') as f:
        model_prosody_config = safe_load(f)

    path_asr = Path(model_asr_config['model_storage']['model_ckpt']).parent
    path_prosody = Path(model_prosody_config['model_storage']['model_ckpt']).parent
    model = Predict_Prosody_Full(model_asr_config["model_setting"],
                                 model_prosody_config["model_setting"],
                                 path_asr,
                                 path_prosody)

    siri_frame_loc = Path(
        main_config['siri_frame'])  # Folder containing pre-calculated siri frames read by name
    stu_mfc_input = Path(main_config['stu_mfc'])

    file_result = []
    for file in stu_mfc_input.glob("**/*.tfs"):
        rel_path = file.relative_to(stu_mfc_input)
        mfc = tf.io.read_file(str(file))
        mfc = tf.io.parse_tensor(mfc, out_type=tf.float32)
        passage_id = file.parent.stem
        siri_frame_file = siri_frame_loc / (passage_id + ".tfs")
        try:
            siri_frame = tf.io.read_file(str(siri_frame_file))
            siri_frame = tf.io.parse_tensor(siri_frame, out_type=tf.float32)
        except Exception as e:
            file_result.append((rel_path, np.nan))
            print(f"Error in file {file}, {e}")
            continue
        # Unstack the siri frame from the first dimension make it iterable
        siri_frame = [x[tf.newaxis, ...] for x in tf.unstack(siri_frame, axis=0)]
        input_data = (mfc, siri_frame)

        try:
            output = model(input_data)
        except Exception as e:
            if not args.skip_err:
                raise e
            print(f"Error in file {file}")
            continue
        #Argmax the output
        output = tf.argmax(output, axis=-1)
        # Offset score from 0~3 to 1~4
        output += 1
        # Convert to numpy and it only have 1 element
        output = output.numpy()[0]
        file_result.append((rel_path, output))
        print(f"File: {file} Predicted: {output}")

    # Create a csv that contain columns: file, predicted
    df = pd.DataFrame(file_result, columns=["file", "predicted"])
    df.to_csv(main_config['output_path'], index=False)
