from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
from DataPipe import DataPipeFactory
import argparse
import sys
import time
from util_function import init_tensorboard, path_resolve


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
    # add name of for current run default is None
    parser.add_argument('--name', type=str, default=None)
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

    # Launching tensorboard
    tb_process = None
    try:
        tb_process = init_tensorboard(log_dir=config['model_storage']['tensorboard_path'], name=args.name)
        print(f"manual debug: tensorboard upload at {config['model_storage']['tensorboard_path']}")
    except Exception as e:
        print(e, file=sys.stderr)
        print("manual debug: tensorboard upload failed")
    # Check if tensorboard is up
    if tb_process.poll() is None:
        print("manual debug: tensorboard is up")
    else:
        print("manual debug: tensorboard is down")
        # print the error message
        tb_process.terminate()
        time.sleep(5)
        tb_process.kill()
        # get communication from the process print error to stderr and output to stdout
        tb_out, tb_err = tb_process.communicate()
        print(tb_out, file=sys.stdout)
        print(tb_err, file=sys.stderr)

    print("manual debug: start training")

    network.fit(dst_train,
                epochs=train_config['epoch'],
                validation_data=dst_test,
                callbacks=[tensorboard_callback, checkpoint_callback, backup_callback])

    # End Tensorboard if tb_process is not None
    if tb_process is not None:
        tb_process.terminate()
        # kill if the process is still alive after 15 seconds
        time.sleep(15)
        if tb_process.poll() is None:
            tb_process.kill()
