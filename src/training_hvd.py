from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
from DataPipe import DataPipeFactory
from pathlib import Path
from shutil import rmtree
import argparse
import horovod.keras as hvd


def unpack(d):
    value_s = d['stu_mfcc']
    start_s = tf.RaggedTensor.from_tensor(d['valid_stu_start'], padding=-1.)
    duration_s = tf.RaggedTensor.from_tensor(d['valid_stu_duration'], padding=-1.)

    # unpack with another key ref_mfcc, valid_ref_start, valid_ref_duration
    value_f = d['ref_mfcc']
    start_f = tf.RaggedTensor.from_tensor(d['valid_ref_start'], padding=-1.)
    duration_f = tf.RaggedTensor.from_tensor(d['valid_ref_duration'], padding=-1.)

    # unpack valid_ref_word

    words = tf.RaggedTensor.from_tensor(d['valid_ref_word'], padding=-1)
    return ((value_s, (start_s, duration_s)), (value_f, (start_f, duration_f))), words


# helper function resolve path related issue it receive a dict return void
def path_resolve(config_dict):
    """
    Sample of configuation file
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
            # make dir and it's parent if exist do nothing
            config_dict['model_storage'][key].mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    # args that if retrain the model default is False
    parser.add_argument('--retrain', type=bool, default=False)
    args = parser.parse_args()
    # load the config
    with open(args.config, 'r') as f:
        config = safe_load(f)
    path_resolve(config)
    print("manual debug: config loaded")
    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # covert all path to string and create the data pipe sample if DataPipeFactory is a class
    data_pipe = DataPipeFactory(config['data_location']['data_record'], config['data_location']['siri_voice'],
                                config['data_location']['siri_meta'], config['cache_location']['cache'])
    print("manual debug: data pipe created")
    # map the data_pipe
    # save the data cache if cache folder is empty
    if not config['cache_location']['cache'].exists():
        print('cache folder is empty, start to save the cache')
        data_pipe.try_save()
        print('cache saved')
    data_pipe.get_raw_data()
    # set the batch size
    config['model_setting']['batch_num'] = config['training_setting']['batch_size']
    # create the network
    network = ASR_Network(**config['model_setting'])
    print("manual debug: network created")
    # create learning rate scheduler
    lr_config = config['training_setting']['learning_rate']
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(lr_config['initial'],
                                                                   lr_config['decay_step'],
                                                                   lr_config['decay'],
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    # add horovod distributed optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)
    # create the optimizer
    callback_config = config['model_storage']
    # create callbacks for tensorboard, checkpoint, and restore
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=callback_config['tensorboard_path'], histogram_freq=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=callback_config['model_ckpt'],
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             monitor='val_loss')
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=callback_config['model_restore'])

    if hvd.rank() == 0:
        callbacks += [tensorboard_callback, checkpoint_callback, backup_callback]
    # train the model
    network.compile(optimizer=optimizer)
    print("manual debug: network compiled")
    train_config = config['training_setting']
    # set datapipe to final state
    dst_train, dst_test = data_pipe.k_fold(total_fold=5,
                                           fold_index=0,
                                           batch_size=train_config['batch_size'],
                                           addition_map=unpack)
    print("manual debug: data pipe set, about to train")
    network.fit(dst_train,
                epochs=train_config['epoch'],
                validation_data=dst_test,
                callbacks=callbacks,
                verbose=1 if hvd.rank() == 0 else 0)
