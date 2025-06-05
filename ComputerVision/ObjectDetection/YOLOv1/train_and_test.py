import silence_tensorflow.auto
import tensorflow as tf
import argparse
import os

from dataset import *
from model import *

def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU to use: -1 for all, 0 for GPU 0, 1 for GPU 1")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--use_backbone", action="store_true", help="Use pretrained backbone")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze pretrained backbone")
    args = parser.parse_args()

    # Memory growth + GPU selection
    physical_gpus = tf.config.list_physical_devices('GPU')
    if args.gpu == -1:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpus_to_use = [f"/GPU:{i}" for i in range(len(physical_gpus))]
    else:
        tf.config.set_visible_devices(physical_gpus[args.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[args.gpu], True)
        gpus_to_use = [f"/GPU:{args.gpu}"]

    strategy = tf.distribute.MirroredStrategy(
        devices=gpus_to_use, cross_device_ops=tf.distribute.NcclAllReduce())

    with strategy.scope():

        if args.use_backbone and not os.path.exists("classifier_backbone.weights.h5"):
            print("Training classifier to get pretrained backbone weights...")
            ds = COCOClassificationDataset()
            train_ds_clf, val_ds_clf = ds.get_data()
            classifier_model = build_classifier()
            classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            classifier_model.fit(train_ds_clf, validation_data=val_ds_clf, epochs=EPOCHS)
            classifier_model.save_weights("checkpoints/classifier_backbone.weights.h5")

        ds = COCODataset()
        train_ds, val_ds = ds.get_data()
        weights = compute_class_weights("../coco/annotations/instances_train2017.json")

        model = build_yolo_v1(
            backbone_weights="checkpoints/classifier_backbone.weights.h5" if args.use_backbone else None,
            freeze_backbone=args.freeze_backbone
        )

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3, first_decay_steps=5000, t_mul=2.0, m_mul=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

        model.compile(optimizer=optimizer,
                      loss=YoloV1Loss(class_weights=weights), metrics=[
            CoordLossMetric(),
            ObjectnessLossMetric(),
            NoObjectnessLossMetric(),
            ClassLossMetric(),
            ClassAccuracy(),
        ])

        if args.resume and os.path.exists(CHECKPOINT_PATH):
            print(f"\nLoading weights from checkpoint: {CHECKPOINT_PATH}")
            model.load_weights(CHECKPOINT_PATH)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               monitor='loss'),
            tf.keras.callbacks.TensorBoard(log_dir="logs")
        ]

        model.fit(train_ds,
                  epochs=EPOCHS,
                  validation_data=val_ds,
                  callbacks=callbacks)

        model.save_weights("yolov1_final_weights.weights.h5")
        print("\nRunning predictions on one batch:")
        for images, _ in train_ds.take(1):
            preds = model.predict(images)
            print("Pred shape:", preds.shape)

if __name__ == "__main__":
    main()
