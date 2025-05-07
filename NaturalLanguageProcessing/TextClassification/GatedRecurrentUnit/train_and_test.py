import tensorflow as tf
import numpy as np
import sys, os
import tqdm

from config import *
from dataset import *
from model import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

def add_sample_weights(features, labels):
    weights = tf.where(labels == 1, 95.6107, 0.5007)
    return features, labels, weights
        
def main():
    data = Dataset()
    train_ds, val_ds = data.get_data()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    print(f"[INFO] Number of devices: {strategy.num_replicas_in_sync}")
    train_ds = train_ds.map(add_sample_weights)
    val_ds = val_ds.map(add_sample_weights)
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)
    
    
    with strategy.scope():
        model = LSTM_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    model.summary(expand_nested=True)
    model.fit(
        train_dist_ds,
        validation_data=val_dist_ds,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2)
        ]
    )
    
if __name__ == '__main__':
    main()
    