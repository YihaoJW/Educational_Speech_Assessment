import sys

sys.path.append('..')

import tensorflow as tf
from pathlib import Path
import argparse
import wandb
from wandb.keras import WandbMetricsLogger
from ProsodyDataPipe import ProsodyDataPipeFactory
from ProsodyPrediction.ProsodyNetwork import ProsodyNetwork
from ProsodyPrediction.ProsodyMetrics import ProsodyLoss, AbsoluteAccuracy, RatingAccuracy, WithInRangeAccuracy, RatingAccuracyDirect, WithInRangeAccuracyDirect, ProsodyKappaLoss
from yaml import safe_load
from util_function import EmergencyExitCallback, EmergencyExit
import tensorflow_models as tfm
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the prosody prediction model")
    # Accpet a yaml file for the configuration
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = safe_load(f)

    student_folder = Path(config['data']['student_folder'])
    assert student_folder.is_dir(), "Student folder is not a directory or not exist"
    siri_folder = Path(config['data']['siri_folder'])
    assert siri_folder.is_dir(), "Siri folder is not a directory or not exist"
    train_record = Path(config['data']['record']) / 'train.tfrecord'
    assert train_record.is_file(), "score file is not a file or not exist"
    eval_record = Path(config['data']['record']) / 'eval.tfrecord'
    assert eval_record.is_file(), "score file is not a file or not exist"
    # Create folder for model storage
    for key in config['model_storage']:
        config['model_storage'][key] = Path(config['model_storage'][key])
        if not config['model_storage'][key].parent.is_dir():
            config['model_storage'][key].parent.mkdir(parents=True, exist_ok=True)
    print("manual debug: Model storage folder created")
    # Create Wandb run and save the run id to tensorboard
    # read run id from file(if exists) else generate a new one and save it to file
    if (config['model_storage']['model_restore'].parent / 'wandb_id.txt').exists():
        with open(config['model_storage']['model_restore'].parent / 'wandb_id.txt', 'r') as f:
            run_id = f.read()
    else:
        run_id = wandb.util.generate_id()
        with open(config['model_storage']['model_restore'].parent / 'wandb_id.txt', 'w') as f:
            f.write(run_id)
    print("manual debug: Run id is", run_id)
    tfb_path = Path(config['model_storage']['tensorboard_path'])
    tfb_path.mkdir(parents=True, exist_ok=True)

    wandb.init(project="ASR_ProsodyPrediction",
               config=config['model_setting'],
               resume="allow", id=run_id,
               dir=tfb_path,
               sync_tensorboard=True)
    print("manual debug: Wandb created")

    # Muti-GPU setting
    strategy = tf.distribute.MirroredStrategy()
    print("manual debug: Strategy created")
    callback_config = config['model_storage']
    lr_config = config['training_setting']['learning_rate']
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=callback_config['tensorboard_path'] / 'tensorboard',
                                                          histogram_freq=5,
                                                          write_graph=False,
                                                          update_freq=25,
                                                          write_steps_per_second=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=callback_config['model_ckpt'],
                                                             save_weights_only=True,
                                                             save_best_only=False,
                                                             mode='max',
                                                             monitor='val_absolute_accuracy')
    backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=callback_config['model_restore'],
                                                          delete_checkpoint=False)
    print("manual debug: Callbacks created")

    with strategy.scope():
        train_factory = ProsodyDataPipeFactory(student_folder, siri_folder, train_record, config["training_setting"]["batch_size"],eval_mode=False)
        eval_factory = ProsodyDataPipeFactory(student_folder, siri_folder, eval_record, 8, eval_mode=False)
        dst = train_factory.get_main_dataset()
        dse = eval_factory.get_main_dataset()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_config['learning_rate'],
                                                                     decay_steps=lr_config['decay_steps'],
                                                                     decay_rate=lr_config['decay_rate'],
                                                                     staircase=True)

        final_lr = tfm.optimization.LinearWarmup(after_warmup_lr_sched=lr_schedule,
                                                 warmup_steps=150,
                                                 warmup_learning_rate=0.0)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True, clipnorm=1.0,clipvalue=0.05)
        # GPU_COUNT = strategy.num_replicas_in_sync
        model = ProsodyNetwork(**config['model_setting'])
        model.compile(optimizer=optimizer,
                      loss=ProsodyLoss(factor=strategy.num_replicas_in_sync),
                      metrics=[AbsoluteAccuracy(name='absolute_accuracy'),
                               RatingAccuracy(name='rating_accuracy'),
                               WithInRangeAccuracy(name='within_range_accuracy'),
                               RatingAccuracyDirect(name='rating_accuracy_direct'),
                               WithInRangeAccuracyDirect(name='within_range_accuracy_direct')])
    attempt = 0
    while True:
        attempt += 1
        try:
            model.fit(dst,
                      epochs=config['training_setting']['epochs'],
                      validation_data=dse,
                      callbacks=[WandbMetricsLogger(log_freq=1),
                                 tensorboard_callback,
                                 checkpoint_callback,
                                 backup_callback,
                                 EmergencyExitCallback(45)])
            wandb.finish()
            break
        except EmergencyExit as e:
            print(f"EmergencyExit occurred during training: {e}", file=sys.stderr)
            print(f"manual debug: EmergencyExit occurred during training: {e}")
            wandb.mark_preempting()
            sys.exit(5)
        except Exception as e:
            print(f"Error occurred during training: {e}", file=sys.stderr)
            print(f"manual debug: Error occurred during training: {e}")
            print(e.with_traceback())

    # run and save evaluation save y_pred and y_true into pair
    y_pred_list = []
    y_true_list = []
    new_dse = ProsodyDataPipeFactory(student_folder, siri_folder, eval_record, 8, eval_mode=True).get_main_dataset()
    for x, y in new_dse:
        y_pred = model.predict(x)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        y_pred_list.append(y_pred)
        y_true_list.append(y)
    # Save the evaluation result to csv
    y_pred = tf.concat(y_pred_list, axis=0)
    y_true = tf.concat(y_true_list, axis=0)
    y_full = tf.concat([y_true, y_pred[..., tf.newaxis]], axis=-1)
    full_df = pd.DataFrame(
        y_full.numpy(),
        columns=['rating', 'rating_min', 'rating_max','rating_pred']
    )
    full_df.to_csv(callback_config['model_restore'].parent / 'evaluation_result.csv', index=True)


