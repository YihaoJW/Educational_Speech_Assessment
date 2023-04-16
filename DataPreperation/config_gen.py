from yaml import safe_dump
import argparse
from pathlib import Path

base_feature = ([None, 80], 512, [128, 256, 384, 512], 3, 2)
base_feature_name = ('input_shape', 'feature_depth', 'channels_list', 'filter_size', 'stack_size')
dense_feature = ([None, 1280], 512, [512, 512], 2)
dense_feature_name = ('input_shape', 'output_shape', 'channels_list', 'stack_size')
word_prediction = ([None, 512], 2500, [512, 1024], 2)
word_prediction_name = ('input_shape', 'output_shape', 'channels_list', 'stack_size')
base_ratio = 125

# generate a dictionary to store the configuration
config = {'model_setting': {'base_feature': dict(zip(base_feature_name, base_feature)),
                            'dense_feature': dict(zip(dense_feature_name, dense_feature)),
                            'word_prediction': dict(zip(word_prediction_name, word_prediction)),
                            'base_ratio': base_ratio},
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add mandatory path that stores the configuration file
    parser.add_argument('--config', type=str, required=True)
    # and data_location that is parent the data location
    parser.add_argument('--data', type=str, required=True)
    # add model storage parent path that stores checkpoint and tensorboard and model restore path
    parser.add_argument('--model', type=str, required=True)
    # add cache location
    parser.add_argument('--cache', type=str, required=True)
    args = parser.parse_args()
    # update all location use pathlib
    data_location_dict = config['data_location']
    # use comprehension to update the dictionary
    data_location_dict = {k: Path(args.data, v).as_posix() for k, v in data_location_dict.items()}
    # use pathlib to update the model storage
    model_storage_dict = config['model_storage']
    model_storage_dict = {k: Path(args.model, v).as_posix() for k, v in model_storage_dict.items()}
    # update the cache location
    cache_loc = config['cache_location']
    cache_loc = {k: Path(args.cache, v).as_posix() for k, v in cache_loc.items()}
    # update the config
    config['data_location'] = data_location_dict
    config['model_storage'] = model_storage_dict
    config['cache_location'] = cache_loc
    # dump the config to the file
    with open(args.config, 'w') as f:
        safe_dump(config, f)
