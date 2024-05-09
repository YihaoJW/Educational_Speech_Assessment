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

    for key in config['model_storage']:
        config['model_storage'][key] = Path(config['model_storage'][key])

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

    train_factory = ProsodyDataPipeFactory(student_folder, siri_folder, train_record, config["training_setting"]["batch_size"],eval_mode=False)
    eval_factory = ProsodyDataPipeFactory(student_folder, siri_folder, eval_record, 8, eval_mode=True)
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
                  loss=ProsodyLoss(factor=8))
    # Load Weights from the backup callback
    last_checkpoint = tf.train.latest_checkpoint(str(callback_config['model_ckpt'].parent))
    if last_checkpoint is not None:
        model.load_weights(last_checkpoint)

    # Iterate though Evaluation dataset manually and save all predictions into csv.
    # Save Y_pred and Y_true into csv
    y_pred_list = []
    y_true_list = []

    for x, y in dse:
        y_pred = model.predict(x)
        y_pred_list.append(y_pred)
        y_true_list.append(y)

    y_pred = tf.concat(y_pred_list, axis=0)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    y_true = tf.concat(y_true_list, axis=0)
    y_full = tf.concat([y_true, y_pred[..., tf.newaxis]], axis=-1)
    full_df = pd.DataFrame(
        y_full.numpy(),
        columns=['rating', 'rating_min', 'rating_max','rating_pred']
    )

    full_df.to_csv(callback_config['model_restore'].parent / 'evaluation_result.csv', index=True)
