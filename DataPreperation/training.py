from yaml import load, dump, safe_load
import tensorflow as tf
from ASR_Network import ASR_Network
from DataPipe import DataPipeFactory
import argparse
import sys
from util_function import path_resolve, EmergencyExit, EmergencyExitCallback, load_config
import wandb
from wandb.keras import WandbMetricsLogger


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
    train_set = DataPipeFactory(train_path, siri_voice, siri_meta, cache / 'train')
    eval_set = DataPipeFactory(eval_path, siri_voice, siri_meta, cache / 'eval')
    return train_set, eval_set


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
    # with open(args.config, 'r') as f:
    #     config = safe_load(f)
    config = load_config(args.config)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # This function will generate all folders if they don't exist. If retrain is True, it will delete the old model.
    # After this function is called, the dir is guaranteed to exist, and the folder is ready for resume or retrain.
    path_resolve(config, args)
    print("manual debug: config loaded")
    # set the batch size
    config['model_setting']['batch_num'] = config['training_setting']['batch_size']

    # Create Wandb run and save the run id to tensorboard
    # read run id from file(if exists) else generate a new one and save it to file
    if (config['model_storage']['model_restore'].parent / 'wandb_id.txt').exists():
        with open(config['model_storage']['model_restore'].parent / 'wandb_id.txt', 'r') as f:
            run_id = f.read()
    else:
        run_id = wandb.util.generate_id()
        with open(config['model_storage']['model_restore'].parent / 'wandb_id.txt', 'w') as f:
            f.write(run_id)

    wandb.init(project="ASR_Model_AttentionBased",
               config=config['model_setting'],
               resume="allow", id=run_id,
               dir=config['model_storage']['model_restore'].parent)
    print("manual debug: Wandb created")
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
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=callback_config['model_restore'],
                                                          delete_checkpoint=False)

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
            dst_train = train_data.get_batch_data(batch_size=train_config['batch_size'], interleave=True, addition_map=unpack)
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

        dst_train = train_data.get_batch_data(batch_size=train_config['batch_size'], interleave=True, addition_map=unpack)
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

    print("manual debug: start training")
    attempt = 0
    while True:
        attempt += 1
        try:
            network.fit(dst_train,
                        epochs=train_config['epoch'],
                        validation_data=dst_test,
                        callbacks=[tensorboard_callback,
                                   checkpoint_callback,
                                   backup_callback,
                                   EmergencyExitCallback(45),
                                   WandbMetricsLogger(log_freq=32)])
            print("manual debug: Training completed successfully.")
            break
        except EmergencyExit as e:
            print(f"EmergencyExit occurred during training: {e}", file=sys.stderr)
            print(f"manual debug: EmergencyExit occurred during training: {e}")
            sys.exit(5)
        except Exception as e:
            print(f"Error occurred during training: {e}", file=sys.stderr)
            print(f"manual debug: the {attempt} failed Retrying training...")

