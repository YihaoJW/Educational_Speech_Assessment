from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
from DataPipe import DataPipeFactory
from pathlib import Path
from shutil import rmtree
import argparse


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
def path_resolve(config_dict, args):
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


def data_train_eval(tf_record_path, siri_voice, siri_meta, cache):
    train_path = tf_record_path.parent / 'Student_Answer_Record_Train.tfrecord'
    assert train_path.exists()
    eval_path = tf_record_path.parent / 'Student_Answer_Record_Eval.tfrecord'
    assert eval_path.exists()
    train = DataPipeFactory(train_path, siri_voice, siri_meta, cache / 'train')

    eval = DataPipeFactory(eval_path, siri_voice, siri_meta, cache / 'eval')
    return train, eval


if __name__ == '__main__':
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    # args that if retrain the model default is False, action is store_true
    parser.add_argument('--retrain', default=False, action='store_true')
    # if you use distributed training default is False action is store_true
    parser.add_argument('--distributed', action='store_true', default=False)
    args = parser.parse_args()
    # load the config
    with open(args.config, 'r') as f:
        config = safe_load(f)
    path_resolve(config, args)
    print("manual debug: config loaded")
    # set the batch size
    config['model_setting']['batch_num'] = config['training_setting']['batch_size']

    print("manual debug: network created")
    # create learning rate scheduler
    lr_config = config['training_setting']['learning_rate']
    callback_config = config['model_storage']
    # create callbacks for tensorboard, checkpoint, and restore
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=callback_config['tensorboard_path'],
                                                          histogram_freq=5,
                                                          write_graph=False,
                                                          write_images=True,
                                                          update_freq=25,
                                                          write_steps_per_second=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=callback_config['model_ckpt'],
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             monitor='val_loss')
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=callback_config['model_restore'])
    # train the model
    train_config = config['training_setting']
    # set datapipe to final state
    if args.distributed:
        print("manual debug: prepare for distributed training")
        strategy = tf.distribute.MirroredStrategy()
        # covert all path to string and create the data pipe sample if DataPipeFactory is a class
        train_data, eval_data = data_train_eval(config['data_location']['data_record'],
                                                config['data_location']['siri_voice'],
                                                config['data_location']['siri_meta'],
                                                config['cache_location']['cache'])
        print("manual debug: data pipe created")
        # map the data_pipe
        # save the data cache if cache folder is empty
        print("manual debug: data pipe save/load start")
        train_data.try_save()
        eval_data.try_save()
        print("manual debug: data pipe save/load end")

        with strategy.scope():
            dst_train = train_data.get_batch_data(batch_size=train_config['batch_size'], addition_map=unpack)
            dst_test = eval_data.get_batch_data(batch_size=train_config['batch_size'], addition_map=unpack)
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(lr_config['initial'],
                                                                           lr_config['decay_step'],
                                                                           lr_config['decay'],
                                                                           staircase=True)
            network = ASR_Network(**config['model_setting'])
            # create the optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, clipnorm=2.0, clipvalue=0.5)
            network.compile(optimizer=optimizer)
    else:
        # covert all path to string and create the data pipe sample if DataPipeFactory is a class
        train_data, eval_data = data_train_eval(config['data_location']['data_record'],
                                                config['data_location']['siri_voice'],
                                                config['data_location']['siri_meta'],
                                                config['cache_location']['cache'])
        print("manual debug: data pipe created")
        # map the data_pipe
        # save the data cache if cache folder is empty
        print("manual debug: data pipe save/load start")
        train_data.try_save()
        eval_data.try_save()
        print("manual debug: data pipe save/load end")

        dst_train = train_data.get_batch_data(batch_size=train_config['batch_size'], addition_map=unpack)
        dst_test = eval_data.get_batch_data(batch_size=train_config['batch_size'], addition_map=unpack)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(lr_config['initial'],
                                                                       lr_config['decay_step'],
                                                                       lr_config['decay'],
                                                                       staircase=True)
        network = ASR_Network(**config['model_setting'])
        # create the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        network.compile(optimizer=optimizer)
    print("manual debug: network compiled")
    print("manual debug: test the network")
    network.evaluate(dst_test)
    print("manual debug: data pipe set, about to train")
    network.fit(dst_train,
                epochs=train_config['epoch'],
                validation_data=dst_test,
                callbacks=[tensorboard_callback, checkpoint_callback, backup_callback])
